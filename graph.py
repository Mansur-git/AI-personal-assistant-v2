from typing import TypedDict, List, Literal,Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_to_dict,
)
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from mem_config import memory
from system_prompts import system_prompts
from pydantic import BaseModel
import tiktoken
import asyncio
import json
import logging
import operator
from logging.handlers import RotatingFileHandler
import functools
import requests
from exa_py import Exa
import os
import time
from bson import ObjectId
from datetime import datetime,timezone
from mongo import(
    Messages,
    Tasks
)
# ---------------- LOGGING SETUP ---------------- #

file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False


def log_call(level=logging.INFO):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                logger.log(level, f"Entering {func.__name__}")
                try:
                    result = await func(*args, **kwargs)
                    logger.log(
                        level,
                        f"Exiting {func.__name__} ({time.time() - start:.3f}s)",
                    )
                    return result
                except Exception:
                    logger.exception(f"Error in {func.__name__}")
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                logger.log(level, f"Entering {func.__name__}")
                try:
                    result = func(*args, **kwargs)
                    logger.log(
                        level,
                        f"Exiting {func.__name__} ({time.time() - start:.3f}s)",
                    )
                    return result
                except Exception:
                    logger.exception(f"Error in {func.__name__}")
                    raise
            return sync_wrapper
    return decorator

# ---------------- CONFIG ---------------- #

load_dotenv()

NUM_MSG_TO_KEEP = 10
MAX_TOKENS = 10000
buffer_lock = asyncio.Lock()
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
BACKGROUND_TASKS = set()

EventType = Literal[
    "preference",
    "behavior",
    "interest",
    "goal",
    "important_people",
    "important_events",
    "other",
]

class State(TypedDict):
    user_id: str
    thread_id:str
    messages: List[BaseMessage]
    turn: int
    total_tokens: int
    summarized_at_msg: int
    total_msg: int
    model: str
    job: str
    summarizing: bool


class MemoryItem(BaseModel):
    store: bool
    event_type: EventType | None = None
    message: str | None = None


class MemoryExtraction(BaseModel):
    items: List[MemoryItem]


# ---------------- UTILITIES ---------------- #
SUMMARY_BUFFER: dict[str, State] = {}
USER_PROFILES: dict[str, list] = {}

def last_message_of_type(messages:List[BaseMessage], msg_type:BaseMessage)->BaseMessage|None:
    try:
        return next((m for m in reversed(messages) if isinstance(m, msg_type)), None)
    except Exception as e:
        logger.error(f"Error in last_message_of_type: {e}", exc_info=True)
        return None

def count_tokens(message: str) -> int:
    """Direct token counting without tool wrapper"""
    try:
        return len(encoding.encode(message))
    except Exception as e:
        logger.error(f"Error in count_tokens: {e}", exc_info=True)
        return 0

# Helper to prevent crashes when dumping MongoDB data
def mongo_serializer(o):
    if isinstance(o, (datetime, ObjectId)):
        return str(o)
    raise TypeError(f"Type {type(o)} not serializable")
# ---------------- LLM SETUP ---------------- #

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------- MEMORY PERSISTENCE ---------------- #


def dump_user_profile_sync(snapshot: dict[str, list]):
    """Dumps buffered user profiles to persistent memory storage"""
    try:
        dumped_count = 0
        for user_id, mem_list in snapshot.items():
            for mem in mem_list:
                try:
                    memory.add(
                        user_id=user_id,
                        messages=mem["message"],
                        metadata={"event_type": mem["event_type"]},
                    )
                    dumped_count += 1
                except Exception:
                    logger.warning(
                        f"Failed to dump memory for user {user_id}: {mem}", exc_info=True
                    )
                    continue
    except Exception as e:
        logger.error(f"Error in dump_user_profile_sync: {e}", exc_info=True)


async def dump_user_profiles(snapshot: dict[str, list]):
    """Dumps buffered user profiles to persistent memory storage"""
    try:
        if not snapshot:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, dump_user_profile_sync, snapshot)
    except Exception as e:
        logger.error(f"Error in dump_user_profiles: {e}", exc_info=True)


async def periodic_memory_dump():
    """Periodically dumps the USER_PROFILE buffer into memory"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            async with buffer_lock:
                snapshot = USER_PROFILES.copy()
                USER_PROFILES.clear()
            await dump_user_profiles(snapshot)
        except Exception as e:
            logger.error(f"Error in periodic_memory_dump: {e}", exc_info=True)

async def memory_extractor(message:str,user_id:str):
    """Extract memories from the last user message"""
    try:
        # bind llm with structured output
        llm_for_memory = llm.with_structured_output(MemoryExtraction)

        if not message:
            return 

        response = await llm_for_memory.ainvoke(
            [
                SystemMessage(content=system_prompts["memory_extraction_prompt"]),
                HumanMessage(content=message),
            ]
        )

        if not response or not response.items:
            return
        for item in response.items:
            try:
                if item.store and item.message and item.message.strip():
                    async with buffer_lock:
                        USER_PROFILES.setdefault(user_id, []).append(
                            {
                                "event_type": item.event_type or "other",
                                "message": item.message,
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to buffer memory item: {item}", exc_info=True)
                continue
    except Exception as e:
        logger.error(f"Error in memory_extractor : {e}", exc_info=True)
#---------------------------DB FUNCTIONS----------------------------#
def make_create_new_task(user_id:str):
    @tool
    async def create_new_task(title:str,description:str,scheduled_for:datetime):
        """Creates a new task and adds it to the database takes title,description and scheduled_for a datetime object"""
        try:
            return  Tasks.create_task(user_id=user_id,title=title,description=description,scheduled_for=scheduled_for)
        except Exception as e:
            logger.error(f"Error occured while creating a task {e}",exc_info=True)
            return "Error occured while creating the task"
    return create_new_task
    
def make_get_task(user_id: str):
    @tool
    async def get_specific_task(task_id:Optional[str]=None,title:Optional[str]=None):
        """Retrieves a specific task by its unique ID."""
        try:
            if task_id:
                return Tasks.get_task(user_id=user_id, task_id=task_id)
            else:
                return Tasks.get_task(user_id=user_id,title=title)
        except Exception as e:
            logger.error(f"Error occurred while retrieving task {task_id}: {e}", exc_info=True)
            return f"Error occurred while retrieving the task: {str(e)}"
    return get_specific_task

def make_get_user_tasks(user_id: str):
    @tool
    async def list_all_tasks():
        """Retrieves all tasks associated with the current user."""
        try:
            return Tasks.get_user_tasks(user_id=user_id)
        except Exception as e:
            logger.error(f"Error occurred while retrieving user tasks: {e}", exc_info=True)
            return "Error occurred while retrieving your tasks"
    return list_all_tasks

def make_get_pending_tasks(user_id: str):
    @tool
    async def list_pending_tasks():
        """Retrieves a list of tasks that are not yet marked as complete."""
        try:
            return Tasks.get_pending_tasks(user_id=user_id)
        except Exception as e:
            logger.error(f"Error occurred while retrieving pending tasks: {e}", exc_info=True)
            return "Error occurred while retrieving pending tasks"
    return list_pending_tasks

def make_get_completed_tasks(user_id: str):
    @tool
    async def list_completed_tasks():
        """Retrieves a list of tasks that have been marked as complete."""
        try:
            return Tasks.get_completed_tasks(user_id=user_id)
        except Exception as e:
            logger.error(f"Error occurred while retrieving completed tasks: {e}", exc_info=True)
            return "Error occurred while retrieving completed tasks"
    return list_completed_tasks

def make_delete_task(user_id: str):
    @tool
    async def delete_existing_task(task_id: str):
        """Permanently removes a task from the database using its ID."""
        try:
            return Tasks.delete_task(user_id=user_id, task_id=task_id)
        except Exception as e:
            logger.error(f"Error occurred while deleting task {task_id}: {e}", exc_info=True)
            return f"Error occurred while deleting the task: {str(e)}"
    return delete_existing_task

def make_update_task(user_id: str):
    @tool
    async def update_existing_task(
        task_id: str, 
        title: Optional[str] = None, 
        description: Optional[str] = None, 
        scheduled_for: Optional[datetime] = None
    ):
        """
        Updates details of an existing task. 
        Provide only the fields that need to be changed (title, description, or scheduled_for).
        """
        try:
            return Tasks.update_task(
                user_id=user_id, 
                task_id=task_id, 
                title=title, 
                description=description, 
                scheduled_for=scheduled_for
            )
        except Exception as e:
            logger.error(f"Error occurred while updating task {task_id}: {e}", exc_info=True)
            return f"Error occurred while updating the task: {str(e)}"
    return update_existing_task

def make_complete_task(user_id: str):
    @tool
    async def mark_task_as_done(task_id: str):
        """Marks a specific task as completed."""
        try:
            return Tasks.complete_task(user_id=user_id, task_id=task_id)
        except Exception as e:
            logger.error(f"Error occurred while completing task {task_id}: {e}", exc_info=True)
            return f"Error occurred while completing the task: {str(e)}"
    return mark_task_as_done

def make_get_overdue_tasks(user_id: str):
    @tool
    async def list_overdue_tasks():
        """Retrieves tasks that are pending and past their scheduled date."""
        try:
            return Tasks.get_overdue_tasks(user_id=user_id)
        except Exception as e:
            logger.error(f"Error occurred while retrieving overdue tasks: {e}", exc_info=True)
            return "Error occurred while retrieving overdue tasks"
    return list_overdue_tasks

def make_get_tasks_by_date_range(user_id: str):
    @tool
    async def list_tasks_in_range(start_date: datetime, end_date: datetime):
        """Retrieves tasks scheduled between a specific start and end datetime."""
        try:
            return Tasks.get_tasks_by_date_range(user_id=user_id, start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.error(f"Error occurred while retrieving tasks by range: {e}", exc_info=True)
            return "Error occurred while retrieving tasks for the specified date range"
    return list_tasks_in_range
#------------------------MESSAGES--------------------------#

def make_add_messages(thread_id: str):
    async def save_chat_message(messages:List[BaseMessage]):
        """Saves a chat message to the database history."""
        try:
            return await Messages.add_messages(thread_id=thread_id,messages=messages)
        except Exception as e:
            logger.error(f"Error occurred while adding message: {e}", exc_info=True)
            return "Error occurred while saving message"
    return save_chat_message

#---------------------------SUMMARY----------------------------------#
async def summarize_memory(snapshot: State):
    """Summarizes messages to reduce memory bloat and cost"""
    try:
        if not snapshot:
            return
        if (
            len(snapshot["messages"]) < NUM_MSG_TO_KEEP + 2
            or snapshot["total_tokens"] < MAX_TOKENS
        ):
            return

        sys_prompt = snapshot["messages"][0]
        old_messages_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in snapshot["messages"][1:-NUM_MSG_TO_KEEP] if m.content)
        summary = await llm.ainvoke(
            [
                SystemMessage(content=system_prompts["summarization_prompt"]),
                HumanMessage(content=old_messages_text),
            ]
        )
        new_messages = (
            [sys_prompt, AIMessage(content=str(summary.content))]
            + snapshot["messages"][-NUM_MSG_TO_KEEP:]
        )
        new_state = snapshot.copy()
        new_state["messages"] = new_messages
        new_state["summarizing"] = False
        new_state["total_msg"] = len(new_state["messages"])
        new_state["total_tokens"] = count_tokens(str(new_messages))
        # Store in summary buffer
        async with buffer_lock:
            SUMMARY_BUFFER[new_state["user_id"]] = new_state
    except Exception as e:
        logger.error(f"Error in summarize_memory: {e}", exc_info=True)
    
# ----------------LLM TOOLS ----------------#


def make_fetch_user_profile(user_id: str):
    @tool
    def fetch_user_profile(query: str) -> List[dict]:
        """Fetches user profile information from memory based on a query, TAKES ONLY ONE ARGUMENT query:str"""
        try:
            if not query:
                return []
            return memory.search(user_id=user_id, query=query, limit=5) or []
        except Exception as e:
            logger.error(f"Error in fetch_user_profile: {e}", exc_info=True)
            return []

    return fetch_user_profile

@tool
def num_tokens(message: str) -> int:
    """Returns the number of tokens in a message"""
    try:
        return len(encoding.encode(message))
    except Exception as e:
        logger.error(f"Error in num_tokens: {e}", exc_info=True)
        return 0

@tool
async def summarize_text(text: str, max_words: int = 150) -> str:
    """Summarizes the given text to the specified max words"""
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=f"Summarize the following text in under {max_words} words."
                ),
                HumanMessage(content=text),
            ]
        )
        return response.content
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}", exc_info=True)
        return "Summary unavailable"

@tool
async def get_weather(city: str) -> str:
    """Fetch the current weather of a city"""

    try:
        url = f"https://wttr.in/{city}?format=%t+%f+%h"

        def fetch():
            return requests.get(url, timeout=10).text.strip()

        weather = await asyncio.to_thread(fetch)

        return f"actual temperature, feels like, humidity: {weather}"

    except Exception:
        logger.exception("Error fetching the weather")
        return "Error fetching the weather"

@tool
async def web_search(query: str) -> str:
    """Search the web and return summarized results"""

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return "Web search unavailable: missing EXA_API_KEY"

    def _search():
        exa = Exa(api_key=api_key)
        return exa.search_and_contents(
            query=query,
            num_results=3,
            text=True,
        )

    try:
        search_response = await asyncio.to_thread(_search)

        if not search_response.results:
            return "No relevant web results found."

        parsed_results = ""
        for i, result in enumerate(search_response.results, 1):
            parsed_results += (
                f"\n[SOURCE {i}]\n"
                f"TITLE: {result.title}\n"
                f"URL: {result.url}\n"
                f"CONTENT:\n{result.text[:1000]}...\n"
            )

        return parsed_results

    except Exception:
        logger.exception("Error occurred during web_search")
        return "Web search failed due to an internal error."


Tools = [num_tokens,summarize_text,get_weather,web_search]
make_tools = [  make_complete_task,
                make_create_new_task,
                make_delete_task,
                make_fetch_user_profile,
                make_get_completed_tasks,
                make_get_overdue_tasks,
                make_get_pending_tasks,
                make_update_task,
                make_get_user_tasks,
                make_get_tasks_by_date_range,
                make_get_task
            ]

# ---------------- GRAPH NODES ---------------- #

async def start_node(state:State):
    available_roles = ["regular"]
    try:
        user_query = input("\nYou>")
        job = await llm.ainvoke([
            SystemMessage(
                content=f"""You are a router based on the user query you have to select one job role from the following 
                {available_roles}
                ###RULES:
                - Reply in ONW WORD ONLY 
                - Your reply must be one of the job roles from the provided list of job roles
                - DO NOT Try to resolve the user query just match the job role according to the user query
                - If you don't understand what job role to chose just select 'regular' 
                """
            ),
            HumanMessage(content=user_query)
        ])
        temp = job.content.strip()
        job = temp if temp in available_roles else 'regular'
        model = 'gpt-4o-mini' if job == 'regular' else 'gpt-4o'

        return {"messages": HumanMessage(content=user_query),
                "model":model,
                "job":job}
    except Exception as e:
        logger.error(f"Error Starting the graph {e}",exc_info=True)
        return {}


async def chat_node(state: State):
    # 1. Setup Tools
    tools = Tools + [t(user_id=state["user_id"]) for t in make_tools]
    TOOL_REGISTRY = {t.name: t for t in tools}
    tool_buffers = {}  
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    context_list = list(state["messages"])
    new_messages = []
    final_text = ""
    first_text= ""
    messages_to_save = [state["messages"][-1]]
    # 2. Iterate through stream (Triggering 'astream_events' in main.py)
    async for chunk in llm_with_tools.astream(context_list):
        # Accumulate text for the final state, but DO NOT yield it here.
        # main.py's 'astream_events' will catch these chunks automatically.
        if chunk.content:
            first_text += chunk.content
        # looking for toolcalls    
        if chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                tool_id = tool_call.get("id")
                if not tool_id:
                    continue  # skip partial calls

                buf = tool_buffers.setdefault(
                    tool_id,
                    {"name": tool_call["name"], "args": {}}
                )

                buf["args"].update(tool_call.get("args", {}))

    if tool_buffers:
        # Build tool_calls list from tool_buffers
        tool_calls_list = [
            {
                "id": tool_id,
                "name": data["name"],
                "args": data["args"],
                "type": "function"
            }
            for tool_id, data in tool_buffers.items()
        ]
        
        new_messages.append(AIMessage(
            content=first_text,
            tool_calls=tool_calls_list
        ))

        final_text += first_text
        first_text = ""
        tool_msg = None
        for tool_id, data in tool_buffers.items():
            tool = TOOL_REGISTRY.get(data["name"])
            if not tool:
                continue

            try:
                result = await tool.ainvoke(data["args"])
                content_str = json.dumps(result, default=mongo_serializer)
            except Exception as tool_err:
                content_str = f"Error executing tool: {tool_err}"

            tool_msg = ToolMessage(
                tool_call_id=tool_id,
                content=content_str,
                name=data["name"]
            )
            new_messages.append(tool_msg)
        if tool_msg:
              async for chunk in llm_with_tools.astream(context_list + new_messages):
                  if chunk.content:
                    first_text += chunk.content

    # 4. Finalize AI Message
    if first_text:
        ai_msg = AIMessage(content=first_text)
        new_messages.append(ai_msg)
    messages_to_save.append(AIMessage(content=final_text + first_text))

    new_msgs = state["messages"] + new_messages
    # 5. Persist 
    try:
        fn = make_add_messages(thread_id=state["thread_id"])
        await fn(messages=messages_to_save)
    except Exception as e:
        logger.error(f"Error saving messages: {e}")

    # 6. RETURN the state update 
    return {"messages":new_msgs}


async def memory_extract_node(state: State):
    """Extract memories from the last user message"""
    try:
        last_user_msg = last_message_of_type(state["messages"],HumanMessage).content
        if last_user_msg:
            last_user_msg = last_user_msg if isinstance(last_user_msg,str) else str(last_user_msg)
            loop = asyncio.get_running_loop()
            def _spawn_memory_task():
                t = asyncio.create_task(
                    memory_extractor(last_user_msg, state["user_id"])
                )
                BACKGROUND_TASKS.add(t)
                t.add_done_callback(BACKGROUND_TASKS.discard)

            loop.call_soon(_spawn_memory_task)

        return {}
    except Exception as e:
        logger.error(f"Error in memory_extract_node: {e}", exc_info=True)
        return {}

async def add_summary(state: State):
    # Schedule summarization (only once)
    if not state.get("summarizing") and state.get("total_tokens", 0) >= MAX_TOKENS:
        try:
            new_messages = []
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage):
                    new_messages.append(
                        ToolMessage(
                            content=msg.content,
                            tool_call_id=msg.tool_call_id,
                        )
                    )
                else:
                    new_messages.append(type(msg)(content=msg.content))
            snapshot = {
                "user_id": state["user_id"],
                "thread_id": state["thread_id"],
                "messages":new_messages,
                "total_tokens": state["total_tokens"],
                "turn": state["turn"],
                "total_msg": state["total_msg"],
                "model": state["model"],
                "job": state["job"],
                "summarized_at_msg": max(
                    0, len(state["messages"]) - NUM_MSG_TO_KEEP
                ),
                "summarizing": True,
            }

            task = asyncio.create_task(summarize_memory(snapshot))
            BACKGROUND_TASKS.add(task)
            task.add_done_callback(BACKGROUND_TASKS.discard)

            return {"summarizing": True}

        except Exception as e:
            logger.error(
                "Failed to schedule summarization task",
                exc_info=True,
            )
            # Do NOT change state on failure
            return {}

    #  Merge summary ONLY if it exists
    if state.get("summarizing"):
        try:
            async with buffer_lock:
                new_state = SUMMARY_BUFFER.pop(state["user_id"], None)

            if not new_state:
                # Summary not ready yet — stay in summarizing mode
                return {}

            start_idx = min(
                new_state.get("summarized_at_msg", 0),
                len(state["messages"]),
            )

            new_state["messages"].extend(state["messages"][start_idx:])
            new_state["total_msg"] = len(new_state["messages"])
            new_state["total_tokens"] = count_tokens(
                str([m.content for m in new_state["messages"]])
            )
            new_state["turn"] = state["turn"]
            new_state["summarizing"] = False

            return new_state

        except Exception as e:
            logger.error(
                "Failed to merge summary state",
                exc_info=True,
            )
            # Stay in summarizing state; retry next turn
            return {}

    return {}


async def update_num_tokens(state: State):
    try:
        # Use direct token counting instead of tool invocation
        token_count = count_tokens("\n".join(m.content for m in state["messages"] if m.content))
        return {
            "total_tokens": token_count,
            "turn": state["turn"] + 1,
            "total_msg": len(state["messages"]),
        }
    except Exception as e:
        logger.error(f"Error in update_num_tokens: {e}", exc_info=True)
        return {}


# ---------------- GRAPH ---------------- #

graph = StateGraph(
    State,
    reducers={
        "messages": operator.add,              # append messages
        "turn": lambda _, new: new,             # overwrite
        "total_tokens": lambda _, new: new,     # overwrite
        "total_msg": lambda _, new: new,        # overwrite
        "summarized_at_msg": lambda _, new: new,
        "summarizing": lambda _, new: new,
        "model": lambda _, new: new,
        "job": lambda _, new: new,
    },
)
graph.add_node("chat_node", chat_node)
graph.add_node("memory_extract_node", memory_extract_node)
graph.add_node("add_summary", add_summary)
graph.add_node("update_num_tokens", update_num_tokens)


graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", "memory_extract_node")
graph.add_edge("memory_extract_node", "add_summary")
graph.add_edge("add_summary", "update_num_tokens")
graph.add_edge("update_num_tokens", END)

app = graph.compile()