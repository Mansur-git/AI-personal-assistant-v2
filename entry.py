from langchain_core.messages import SystemMessage, HumanMessage
from graph import app
import asyncio
from system_prompts import system_prompts
from mongo import get_checkpoint,upsert_checkpoint
from mem_config import memory 
from graph import periodic_memory_dump
from graph import State
from startup import startup


#creating initial state
def create_initial_state(
    user_id: str,
    job: str = "regular",
    model: str = "gpt-4o-mini",
    ) -> State:
    return State(
        user_id=user_id,
        messages=[SystemMessage(content=system_prompts[job])],
        thread_id=thread_id,
        turn=0,
        job=job,
        model=model,
        total_tokens=0,
        total_msg=0,
        summarized_at_msg=0,
        summarizing=False,
    )

async def main():
    # start periodic memory dump task
    asyncio.create_task(periodic_memory_dump())
    startup()
    # Attempt to load existing state from Mongo first
    loaded_state = await get_checkpoint(thread_id=thread_id)
    
    if loaded_state:
        state = loaded_state
        print("\n--- Resuming Session ---")
    else:
        state = create_initial_state(user_id=user_id,job='regular',model='gpt-4o-mini')
        print("\nHello I am Yaara!, I am happy to assist you")

    try:
        while True:
            user_query = input('\nYou> ')
            state["messages"].append(HumanMessage(content=user_query))
            
            # Run the graph
            state = await app.ainvoke(state) 
            
            # Save 
            await upsert_checkpoint(thread_id=thread_id,state=state)
            
    except KeyboardInterrupt:
        print("Session ended")
    
asyncio.run(main())