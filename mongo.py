from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError, DuplicateKeyError
# from bson.errors import InvalidId
from dotenv import load_dotenv
import os
from typing import  Optional, List, Dict, Any,Annotated,TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime, timezone
from bson import ObjectId
import logging
from langchain_core.messages import BaseMessage
from logging.handlers import RotatingFileHandler
import re
load_dotenv()

#-----------------LOGGER------------------#
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

#---------------------------STATE-----------------#

class State(TypedDict):
    user_id: str
    thread_id:str
    messages: Annotated[List[BaseMessage], add_messages]
    turn: int
    total_tokens: int
    summarized_at_msg: int
    total_msg: int
    model: str
    job: str
    summarizing: bool


# MongoDB client setup (Async)
# Motor client is asynchronous
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]

# ============================================================================
# USER FUNCTIONS
# ============================================================================
class Users:
    async def create_user(username: str, email: str, password_hash: str) -> Optional[ObjectId]:
        """Create a new user"""
        try:
            result = await db.users.insert_one({
                "username": username,
                "email": email,
                "password": password_hash,
                "created_at": datetime.now(timezone.utc)  
            })
            return result.inserted_id
        except DuplicateKeyError:
            logger.error(f"Error creating user: Email {email} already exists")
            return None
        except PyMongoError as e:
            logger.error(f"Database error creating user: {e}")
            return None

    async def get_user(user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            if not ObjectId.is_valid(user_id):
                logger.error(f"Invalid user_id format: {user_id}")
                return None
            return await db.users.find_one({"_id": ObjectId(user_id)})
        except PyMongoError as e:
            logger.error(f"Database error getting user {user_id}: {e}")
            return None

    async def get_user_by_email(email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            return await db.users.find_one({"email": email})
        except PyMongoError as e:
            logger.error(f"Database error getting user by email {email}: {e}")
            return None

    async def get_user_by_username(username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            return await db.users.find_one({"username": username})
        except PyMongoError as e:
            logger.error(f"Database error getting user by username {username}: {e}")
            return None

    async def update_user(user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user information"""
        try:
            if not ObjectId.is_valid(user_id):
                return False
                
            result = await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {**update_data, "updated_at": datetime.now(timezone.utc)}}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Database error updating user {user_id}: {e}")
            return False

    async def delete_user(user_id: str) -> bool:
        """Delete user and all associated data"""
        try:
            if not ObjectId.is_valid(user_id):
                return False
                
            user_obj_id = ObjectId(user_id)
            
            # Get all threads for this user
            # In Motor, find returns a cursor, use to_list
            cursor = db.threads.find({"user_id": user_obj_id})
            threads = await cursor.to_list(length=None)
            
            # Delete all associated data
            for thread in threads:
                thread_id = thread["_id"]
                await db.messages.delete_one({"thread_id": thread_id})
                await db.checkpoints.delete_one({"_id": thread_id})
            
            # Delete threads and tasks
            await db.threads.delete_many({"user_id": user_obj_id})
            await db.tasks.delete_many({"user_id": user_obj_id})
            
            # Delete user
            result = await db.users.delete_one({"_id": user_obj_id})
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Database error deleting user {user_id}: {e}")
            return False

    async def is_valid_user_id(user_id: str) -> bool:
        if not ObjectId.is_valid(user_id):
            return False
        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            return user is not None
        except Exception as e:
            logger.error(f"Database error validating user id: {e}")

# ============================================================================
# THREAD FUNCTIONS
# ============================================================================
class Threads:
    async def create_thread(user_id: str) -> Optional[ObjectId]:
        """Create a new thread for a user"""
        try:
            if not ObjectId.is_valid(user_id):
                return None
                
            result = await db.threads.insert_one({
                "user_id": ObjectId(user_id),
                "name": "New thread",
                "created_at": datetime.now(timezone.utc)
            })
            thread_id = result.inserted_id
            await Messages.init_messages(thread_id)  # Initialize messages for this thread
            return thread_id
        except PyMongoError as e:
            logger.error(f"Database error creating thread for user {user_id}: {e}")
            return None

    async def get_thread(thread_id: str) -> Optional[Dict]:
        """Get thread by ID"""
        try:
            if not ObjectId.is_valid(thread_id):
                return None
            return await db.threads.find_one({"_id": ObjectId(thread_id)})
        except PyMongoError as e:
            logger.error(f"Database error getting thread {thread_id}: {e}")
            return None

    async def get_user_threads(user_id: str) -> List[Dict]:
        """Get all threads for a user"""
        try:
            if not ObjectId.is_valid(user_id):
                return []
            
            cursor = db.threads.find({"user_id": ObjectId(user_id)}).sort("created_at", -1)
            threads = await cursor.to_list(length=None)
            
            # FIX: Normalize thread_id field for frontend consistency
            for thread in threads:
                thread["thread_id"] = str(thread["_id"])
            
            return threads
        except PyMongoError as e:
            logger.error(f"Database error getting threads for user {user_id}: {e}")
            return []

    async def delete_thread(thread_id: str) -> None:
        """Delete a thread and its associated data"""
        try:
            if not ObjectId.is_valid(thread_id):
                return

            thread_obj_id = ObjectId(thread_id)
            await db.threads.delete_one({"_id": thread_obj_id})
            await db.messages.delete_one({"thread_id": thread_obj_id})
            await db.checkpoints.delete_one({"_id": thread_obj_id})
        except PyMongoError as e:
            logger.error(f"Database error deleting thread {thread_id}: {e}")

# ============================================================================
# MESSAGE FUNCTIONS
# ============================================================================
class Messages:
    async def init_messages(thread_id: ObjectId) -> None:
        """Initialize message history for a thread"""
        try:
            await db.messages.insert_one({
                "thread_id": thread_id,
                "history": [],
                "created_at": datetime.now(timezone.utc)
            })
        except PyMongoError as e:
            logger.error(f"Database error initializing messages for thread {thread_id}: {e}")

    async def add_messages(thread_id: str, messages: List[BaseMessage]) -> None:
        """
        Saves only HUMAN and AI messages for the UI/Display log.
        Ignores ToolMessages, SystemMessages, and empty AI tool-call requests.
        """
        try:
            if not ObjectId.is_valid(thread_id):
                return

            messages_to_push = []
            
            for msg in messages:
                # 1. Filter by Type: Only keep conversation
                if msg.type not in ['human', 'ai']:
                    continue
                
                if not msg.content:
                    continue

                doc = {
                    "role": msg.type, 
                    "content": msg.content,
                    "timestamp": datetime.now(timezone.utc)
                }
                messages_to_push.append(doc)

            if not messages_to_push:
                return

            # 3. Batch Insert
            await db.messages.update_one(
                {"thread_id": ObjectId(thread_id)},
                {
                    "$push": {
                        "history": {
                            "$each": messages_to_push
                        }
                    }
                },
                upsert=True
            )
        except PyMongoError as e:
            logger.error(f"Database error adding messages to thread {thread_id}: {e}")

    async def get_messages(thread_id: str) -> List[BaseMessage]:
        """Get all messages for a thread and convert them to LangChain BaseMessage objects"""
        try:
            if not ObjectId.is_valid(thread_id):
                return []

            result = await db.messages.find_one({"thread_id": ObjectId(thread_id)})
            
            if not result or "history" not in result:
                return []
            
            # FIX: Convert MongoDB documents back to LangChain messages
            from langchain_core.messages import HumanMessage, AIMessage
            
            converted_messages = []
            for msg in result["history"]:
                if msg["role"] == "human":
                    converted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    converted_messages.append(AIMessage(content=msg["content"]))
            
            return converted_messages
            
        except PyMongoError as e:
            logger.error(f"Database error getting messages for thread {thread_id}: {e}")
            return []

    async def clear_messages(thread_id: str) -> bool:
        """Clear all messages in a thread"""
        try:
            if not ObjectId.is_valid(thread_id):
                return False

            result = await db.messages.update_one(
                {"thread_id": ObjectId(thread_id)},
                {"$set": {"history": []}}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Database error clearing messages for thread {thread_id}: {e}")
            return False

    async def delete_last_message(thread_id: str) -> bool:
        """Delete the last message in a thread"""
        try:
            if not ObjectId.is_valid(thread_id):
                return False

            result = await db.messages.update_one(
                {"thread_id": ObjectId(thread_id)},
                {"$pop": {"history": 1}}  # Remove last element
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Database error deleting last message for thread {thread_id}: {e}")
            return False

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
class CheckPoints:
    async def upsert_checkpoint(thread_id: str, state: State) -> None:
        """Create or update checkpoint state for a thread"""
        try:
            if not ObjectId.is_valid(thread_id):
                return

            await db.checkpoints.update_one(
                {"_id": ObjectId(thread_id)},
                {"$set": {
                    "state": state,
                    "updated_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )
        except PyMongoError as e:
            logger.error(f"Database error upserting checkpoint for thread {thread_id}: {e}")

    async def get_checkpoint(thread_id: str) -> Optional[Dict]:
        """Get checkpoint state for a thread"""
        try:
            if not ObjectId.is_valid(thread_id):
                return None

            result = await db.checkpoints.find_one({"_id": ObjectId(thread_id)})
            return result["state"] if result else None
        except PyMongoError as e:
            logger.error(f"Database error getting checkpoint for thread {thread_id}: {e}")
            return None

    async def delete_checkpoint(thread_id: str) -> bool:
        """Delete checkpoint for a thread"""
        try:
            if not ObjectId.is_valid(thread_id):
                return False

            result = await db.checkpoints.delete_one({"_id": ObjectId(thread_id)})
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Database error deleting checkpoint for thread {thread_id}: {e}")
            return False

# ============================================================================
# TASK FUNCTIONS
# ============================================================================
class Tasks:
    async def create_task(
        user_id: str,
        title: str,
        description: str,
        scheduled_for: datetime
    ) -> Optional[ObjectId]:
        """Create a new task"""
        try:
            if not ObjectId.is_valid(user_id):
                logger.error(f"Invalid user_id: {user_id}")
                return None

            result = await db.tasks.insert_one({
                "user_id": ObjectId(user_id),
                "title": title,
                "description": description,
                "created_at": datetime.now(timezone.utc),
                "scheduled_for": scheduled_for,
                "status": "PENDING",
                "edited": False
            })
            return result.inserted_id
        except PyMongoError as e:
            logger.error(f"Database error creating task for user {user_id}: {e}")
            return None

    async def get_task(user_id: str, task_id: Optional[str] = None, title: Optional[str] = None) -> Optional[Dict]:
        """
        Get a task by either its ID or its exact Title.
        Must provide at least one of task_id or title.
        """
        try:
            # FIX: Convert user_id to ObjectId
            if not ObjectId.is_valid(user_id):
                logger.error(f"Invalid user_id: {user_id}")
                return None
                
            query = {"user_id": ObjectId(user_id)}

            if task_id:
                if not ObjectId.is_valid(task_id):
                    return None
                query["_id"] = ObjectId(task_id)
                
            elif title:
                # Case-insensitive match for convenience
                query["title"] = {"$regex": f"^{re.escape(title)}$", "$options": "i"}
                
            else:
                # No identifier provided
                return None

            return await db.tasks.find_one(query)

        except PyMongoError as e:
            logger.error(f"Database error getting task: {e}")
            return None

    async def get_user_tasks(user_id: str, status: Optional[str] = None) -> List[Dict]:
        """Get all tasks for a user, optionally filtered by status"""
        try:
            if not ObjectId.is_valid(user_id):
                return []

            query = {"user_id": ObjectId(user_id)}
            if status:
                query["status"] = status
            
            cursor = db.tasks.find(query).sort("scheduled_for", 1)
            return await cursor.to_list(length=None)
        except PyMongoError as e:
            logger.error(f"Database error getting tasks for user {user_id}: {e}")
            return []

    async def update_task(task_id: str, update_data: Dict[str, Any]) -> bool:
        """Update task information"""
        try:
            if not ObjectId.is_valid(task_id):
                return False

            update_data["edited"] = True
            update_data["updated_at"] = datetime.now(timezone.utc)
            
            result = await db.tasks.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Database error updating task {task_id}: {e}")
            return False

    async def complete_task(task_id: str) -> bool:
        """Mark a task as completed"""
        try:
            if not ObjectId.is_valid(task_id):
                return False

            result = await db.tasks.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": {
                    "status": "COMPLETED",
                    "completed_at": datetime.now(timezone.utc)
                }}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Database error completing task {task_id}: {e}")
            return False

    async def delete_task(task_id: str) -> bool:
        """Delete a task"""
        try:
            if not ObjectId.is_valid(task_id):
                return False

            result = await db.tasks.delete_one({"_id": ObjectId(task_id)})
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Database error deleting task {task_id}: {e}")
            return False

    async def get_pending_tasks(user_id: str) -> List[Dict]:
        """Get all pending tasks for a user"""
        return await get_user_tasks(user_id, status="PENDING")

    async def get_completed_tasks(user_id: str) -> List[Dict]:
        """Get all completed tasks for a user"""
        return await get_user_tasks(user_id, status="COMPLETED")

    async def get_tasks_by_date_range(user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get tasks scheduled within a date range"""
        try:
            if not ObjectId.is_valid(user_id):
                return []

            cursor = db.tasks.find({
                "user_id": ObjectId(user_id),
                "scheduled_for": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }).sort("scheduled_for", 1)
            return await cursor.to_list(length=None)
        except PyMongoError as e:
            logger.error(f"Database error getting tasks by date range for user {user_id}: {e}")
            return []

    async def get_overdue_tasks(user_id: str) -> List[Dict]:
        """Get all overdue pending tasks"""
        try:
            if not ObjectId.is_valid(user_id):
                return []

            cursor = db.tasks.find({
                "user_id": ObjectId(user_id),
                "status": "PENDING",
                "scheduled_for": {"$lt": datetime.now(timezone.utc)}
            }).sort("scheduled_for", 1)
            return await cursor.to_list(length=None)
        except PyMongoError as e:
            logger.error(f"Database error getting overdue tasks for user {user_id}: {e}")
            return []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def create_indexes():
    """Create indexes for better query performance"""
    try:
        # User indexes
        await db.users.create_index("email", unique=True)
        #await db.users.create_index("username", unique=True)
        
        # Thread indexes
        await db.threads.create_index("user_id")
        
        # Message indexes
        await db.messages.create_index("thread_id", unique=True)
        
        # Task indexes
        await db.tasks.create_index([("user_id", 1), ("status", 1)])
        await db.tasks.create_index([("user_id", 1), ("scheduled_for", 1)])
        
        logger.info("Database indexes created successfully")
    except PyMongoError as e:
        logger.error(f"Database error creating indexes: {e}")

def close_connection():
    """Close MongoDB connection"""
    # MotorClient.close() is synchronous
    try:
        client.close()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")