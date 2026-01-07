from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from mem_config import memory
from system_prompts import system_prompts
from pydantic import BaseModel
import tiktoken
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import time
import functools

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

NUM_MSG_TO_KEEP = 5
MAX_TOKENS = 1000
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

def last_message_of_type(messages, msg_type):
    try:
        return next((m for m in reversed(messages) if isinstance(m, msg_type)), None)
    except Exception as e:
        logger.error(f"Error in last_message_of_type: {e}", exc_info=True)
        return None

def messages_to_text(messages: List[BaseMessage]) -> str:
    """Converts a list of messages to a plain text representation, including tool calls."""
    try:
        lines = []
        for m in messages:
            # Handle standard content
            text = f"{type(m).__name__}: {m.content}"
            
            # Handle Tool Calls (The missing piece)
            if isinstance(m, AIMessage) and m.tool_calls:
                tools_desc = ", ".join(
                    f"{t['name']}({t['args']})" for t in m.tool_calls
                )
                text += f" [Tool Calls: {tools_desc}]"
                
            lines.append(text)
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in messages_to_text: {e}", exc_info=True)
        return ""

def count_tokens(message: str) -> int:
    """Direct token counting without tool wrapper"""
    try:
        return len(encoding.encode(message))
    except Exception as e:
        logger.error(f"Error in count_tokens: {e}", exc_info=True)
        return 0


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


async def summarize_memory(snapshot: State):
    """Summarizes messages to reduce memory bloat and cost"""
    logger.log("summarizing\n")
    try:
        if not snapshot:
            return
        if (
            len(snapshot["messages"]) < NUM_MSG_TO_KEEP + 2
            or snapshot["total_tokens"] < MAX_TOKENS
        ):
            return

        sys_prompt = snapshot["messages"][0]
        old_messages_text = messages_to_text(snapshot["messages"][1:-NUM_MSG_TO_KEEP])
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
        new_state = {
            "user_id": snapshot["user_id"],
            "messages": new_messages,
            "total_tokens": count_tokens(messages_to_text(new_messages)),
            "turn": snapshot["turn"],
            "total_msg": len(new_messages),
            "model": snapshot["model"],
            "job": snapshot["job"],
            "summarized_at_msg": snapshot["total_msg"],
            "summarizing": False
        }
        # Store in summary buffer
        async with buffer_lock:
            SUMMARY_BUFFER[new_state["user_id"]] = new_state
    except Exception as e:
        logger.error(f"Error in summarize_memory: {e}", exc_info=True)

# ---------------- TOOLS ---------------- #


@log_call()
def make_fetch_user_profile(user_id: str):
    @tool
    def fetch_user_profile(query: str) -> List[dict]:
        """Fetches user profile information from memory based on a query"""
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


# ---------------- GRAPH NODES ---------------- #


@log_call()
async def chat_node(state: State):
    try:
        # create tool instances with bound user_id
        tools = [make_fetch_user_profile(state["user_id"]), num_tokens, summarize_text]
        TOOL_REGISTRY = {t.name: t for t in tools}

        # bind tools to llm
        llm_with_tools = llm.bind_tools(tools)
        messages = list(state["messages"])
        # for tool in tools:
        #     logger.info(f"Tool name: {tool.name}")
        #     logger.info(f"Tool description: {tool.description}")
        #     logger.info(f"Tool args schema: {tool.args_schema.schema() if hasattr(tool, 'args_schema') else 'No schema'}")
        
        while True:
            response = await llm_with_tools.ainvoke(messages)

            if not response.tool_calls:
                print(f"\nYaara: {response.content}\n")
                messages.append(response)
                break

            messages.append(response)

            for tool_call in response.tool_calls:
                try:
                    tool = TOOL_REGISTRY.get(tool_call["name"])
                    if not tool:
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="Unknown tool requested",
                            )
                        )
                        continue

                    if asyncio.iscoroutinefunction(tool.func):
                        result = await tool.func(**tool_call["args"])
                    else:
                        result = tool.func(**tool_call["args"])

                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=json.dumps(result)
                            if not isinstance(result, str)
                            else result,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error executing tool {tool_call.get('name')}: {e}", exc_info=True)
                    messages.append(
                        ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {str(e)}")
                    )

        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error in chat_node: {e}", exc_info=True)
        return {"messages": state["messages"]}


@log_call()
async def memory_extract_node(state: State):
    """Extract memories from the last user message"""
    try:
        # bind llm with structured output
        llm_for_memory = llm.with_structured_output(MemoryExtraction)

        last_user = last_message_of_type(state["messages"], HumanMessage)
        if not last_user:
            return {}

        response = await llm_for_memory.ainvoke(
            [
                SystemMessage(content=system_prompts["memory_extraction_prompt"]),
                HumanMessage(content=last_user.content),
            ]
        )

        if not response or not response.items:
            return {}

        for item in response.items:
            try:
                if item.store and item.message and item.message.strip():
                    async with buffer_lock:
                        USER_PROFILES.setdefault(state["user_id"], []).append(
                            {
                                "event_type": item.event_type or "other",
                                "message": item.message,
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to buffer memory item: {item}", exc_info=True)
                continue

        return {}
    except Exception as e:
        logger.error(f"Error in memory_extract_node: {e}", exc_info=True)
        return {}

@log_call()
async def add_summary(state: State):
    # Schedule summarization (only once)
    if not state.get("summarizing") and state.get("total_tokens", 0) >= 1000:
        try:
            snapshot = {
                "user_id": state["user_id"],
                "messages": list(state["messages"]),
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
                messages_to_text(new_state["messages"])
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



@log_call()
async def update_num_tokens(state: State):
    try:
        # Use direct token counting instead of tool invocation
        token_count = count_tokens(messages_to_text(state["messages"]))
        return {
            "total_tokens": token_count,
            "turn": state["turn"] + 1,
            "total_msg": len(state["messages"]),
        }
    except Exception as e:
        logger.error(f"Error in update_num_tokens: {e}", exc_info=True)
        return {}


# ---------------- GRAPH ---------------- #

graph = StateGraph(State)
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