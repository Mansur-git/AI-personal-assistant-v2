from langchain_core.messages import SystemMessage, HumanMessage
from graph import app
import asyncio
from system_prompts import system_prompts
from mongo import load_state, save_state
from mem_config import memory 
from graph import periodic_memory_dump
from graph import State
from graph import add_summary
SESSION_ID = '123' 

async def main():
    # start periodic memory dump task
    asyncio.create_task(periodic_memory_dump())

    # Attempt to load existing state from Mongo first
    loaded_state = load_state(SESSION_ID)
    
    if loaded_state:
        state = loaded_state
        print("\n--- Resuming Session ---")
    else:
        state = {
            "user_id":"mansur",
            "messages": [
                SystemMessage(content=system_prompts['regular']),
            ],
            "turn": 0,
            "job": "regular",
            "model":"gpt-4o-mini",
            "total_tokens": 0,
            "total_msg":0,
            "summarized_at_msg":0,
            "summarizing": False
        }
        print("\nHello I am Yaara!, I am happy to assist you")

    try:
        while True:
            user_query = input('\nYou> ')
            state["messages"].append(HumanMessage(content=user_query))
            
            # Run the graph
            state = await app.ainvoke(state) 
            
            # Save 
            save_state(SESSION_ID, state)
            
    except KeyboardInterrupt:
        print("Session ended")
    
asyncio.run(main())