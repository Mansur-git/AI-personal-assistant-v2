import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    message_to_dict,
    messages_to_dict,
    messages_from_dict,
    _message_from_dict
)
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- Database & Security Imports ---
from mongo import (
    Users, 
    Threads, 
    CheckPoints, 
    Messages,
    create_indexes
)
import json
from security import hash_password, verify_password
from jwt_utils import create_access_token, get_current_user

# --- Graph & Logic Imports ---
from langchain_core.messages import SystemMessage, HumanMessage
from graph import app as graph_app, State, periodic_memory_dump, BACKGROUND_TASKS
from system_prompts import system_prompts

# -------------------------- LOGGING CONFIG -------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yaara_backend")

# -------------------------- LIFESPAN MANAGER -------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Starting up... ---")
    await create_indexes()
    dump_task = asyncio.create_task(periodic_memory_dump())
    BACKGROUND_TASKS.add(dump_task)
    yield
    logger.info("--- Shutting down... ---")
    dump_task.cancel()
    try:
        await dump_task
    except asyncio.CancelledError:
        pass
    BACKGROUND_TASKS.discard(dump_task)

# -------------------------- APP INITIALIZATION -------------------------- #

app = FastAPI(lifespan=lifespan, title="Yaara API", version="1.0.0")

# -------------------------- CORS CONFIGURATION -------------------------- #

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- HELPER FUNCTIONS -------------------------- #
                     
def create_initial_state(
    user_id: str,
    thread_id: str,
    job: str = "regular",
    model: str = "gpt-4o-mini",
) -> State:
    return State(
        user_id=user_id,
        messages=[SystemMessage(content=system_prompts.get(job, system_prompts["regular"]))],
        thread_id=thread_id,
        turn=0,
        job=job,
        model=model,
        total_tokens=0,
        total_msg=0,
        summarized_at_msg=0,
        summarizing=False,
    )

async def verify_thread_access(thread_id: str, user_id: str):
    """
    Security Check: Ensures the thread exists and belongs to the requesting user.
    """
    threads = await Threads.get_user_threads(user_id)
    # FIX: Use the normalized thread_id field
    is_owned = any(t.get("thread_id") == thread_id for t in threads)
    
    if not is_owned:
        logger.warning(f"Unauthorized access attempt: User {user_id} tried to access Thread {thread_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="You do not have permission to access this thread."
        )

# -------------------------- AUTH ENDPOINTS -------------------------- #

class AuthRequest(BaseModel):
    email: str
    password: str
    user_name: Optional[str] = None

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(data: AuthRequest):
    if await Users.get_user_by_email(email=data.email):
        raise HTTPException(status_code=400, detail="User already exists")

    await Users.create_user(
        username=data.user_name,
        email=data.email,
        password_hash=hash_password(data.password)
    )
    return {"status": "created"}

@app.post("/auth/login")
async def login(data: AuthRequest):
    user = await Users.get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(str(user["_id"]))
    return {"access_token": token}

# -------------------------- THREAD ENDPOINTS -------------------------- #

@app.post("/thread/create", status_code=status.HTTP_201_CREATED)
async def create_thread(user_id: str = Depends(get_current_user)):
    new_thread_id = await Threads.create_thread(user_id=user_id)
    logger.info(f"Thread created: {new_thread_id} for user {user_id}")
    return {
        "status": "created",
        "thread_id": str(new_thread_id)
    }

@app.get("/threads")
async def list_threads(user_id: str = Depends(get_current_user)):
    return await Threads.get_user_threads(user_id)

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    await verify_thread_access(thread_id, user_id)
    await Threads.delete_thread(thread_id)
    logger.info(f"Thread {thread_id} deleted by user {user_id}")
    return {"deleted": True}

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str, user_id: str = Depends(get_current_user)):
    # 1. Security Check
    await verify_thread_access(thread_id, user_id)

    # 2. Fetch from DB
    # Assuming get_messages returns a list of LangChain message objects or dicts
    messages = await Messages.get_messages(thread_id=thread_id)

    # 3. Format for Frontend
    # We need to convert LangChain/DB format to simple JSON: { role, content }
    formatted_messages = []
    for msg in messages:
        # Handle if msg is a dict (from DB) or Object (from LangChain)
        role = getattr(msg, 'type', None) or msg.get('type') or msg.get('role')
        content = getattr(msg, 'content', None) or msg.get('content')
        
        # Map 'human' -> 'user' for your frontend
        if role == 'human':
            role = 'user'
            
        formatted_messages.append({
            "role": role,
            "content": content
        })

    return formatted_messages


# -------------------------- CHAT ENDPOINTS -------------------------- #

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

@app.post("/chat")
async def send_message(
    data: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    async def event_generator():
        try:
            print(f"--- STARTING STREAM for Thread {data.thread_id} ---") # DEBUG
            
            # ... (Your existing Thread/Checkpoint loading logic here) ...
            # Assume thread_id and state are loaded correctly
            
            if data.thread_id:
                await verify_thread_access(data.thread_id, user_id)
                thread_id = data.thread_id
            else:
                thread_id = str(await Threads.create_thread(user_id=user_id))
            
            checkpoint = await CheckPoints.get_checkpoint(thread_id = thread_id)

            if checkpoint:
                new_messages = messages_from_dict(checkpoint["messages"]) 
                checkpoint["messages"] = new_messages
                state = checkpoint 
            else:
                state = create_initial_state(user_id=user_id, thread_id=thread_id)
            state["user_id"] = user_id
            state["messages"].append(HumanMessage(content=data.message))

            # Flag to track if we actually saved
            checkpoint_saved = False 

            # EVENT LOOP
            async for event in graph_app.astream_events(state, version="v1"):
                
                # DEBUG: Print every single event type to see what is happening
                kind = event["event"]
                print(f"Received Event: {kind}") 

                # 1. STREAMING (Try both event types)
                # 'on_chat_model_stream' is often more reliable than 'on_llm_stream' in newer versions
                if kind == "on_llm_stream" or kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(f"Yielding chunk: {chunk.content[:10]}...") # DEBUG
                        yield f"data: {chunk.content}\n\n"

                # 2. FINISHING
                elif kind == "on_chain_end":
                    print("--- GRAPH END DETECTED ---") # DEBUG
                    
                    
                    if state:
                        # Extract messages safely
                        
                        new_messages = messages_to_dict(state.get("messages",[]))
                        snapshot = {
                            "user_id": state.get("user_id"),
                            "thread_id":state.get("thread_id"),
                            "messages": new_messages,
                            # Use .get() to prevent KeyErrors
                            "total_tokens": state.get("total_tokens", 0),
                            "turn": state.get("turn", 0),
                            "model": state.get("model", ""),
                            "summarizing":state.get("summarizing",False),
                            "job":state.get("job"),
                            "total_msg":state.get("total_msg"),
                            "summarized_at_msg":state.get("summarized_at_msg")
                        }

                        print(f"Saving Checkpoint for {thread_id}...") # DEBUG
                        await CheckPoints.upsert_checkpoint(thread_id=thread_id, state=snapshot)
                        checkpoint_saved = True
                        print("Checkpoint Saved Successfully") # DEBUG
            
            if not checkpoint_saved:
                print("WARNING: Loop finished but checkpoint was NOT saved.")

            yield "data: [DONE]\n\n"

        except Exception as e:
            # This catches code errors and prints them to your server console
            import traceback
            traceback.print_exc()
            logger.error(f"Chat stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
