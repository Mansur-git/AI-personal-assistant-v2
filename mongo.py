from pymongo import MongoClient
from dotenv import load_dotenv
import os
from langchain_core.messages import message_to_dict

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]

sessions = db.sessions

from langchain_core.messages import message_to_dict, messages_from_dict

def save_state(session_id, state):
    # Create a copy so we don't modify the running 'state' variable
    serializable_state = state.copy()
    
    # Convert the list of Message objects into a list of plain dictionaries
    serializable_state['messages'] = [message_to_dict(m) for m in state['messages']]
    
    # Now MongoDB will accept it
    sessions.update_one(
        {"session_id": session_id},
        {"$set": {"state": serializable_state}},
        upsert=True
    )
    
def load_state(session_id):
    doc = sessions.find_one({"session_id": session_id})
    if doc:
        state = doc['state']
        # Convert dictionaries back to SystemMessage/HumanMessage objects
        state['messages'] = messages_from_dict(state['messages'])
        return state
    return None

