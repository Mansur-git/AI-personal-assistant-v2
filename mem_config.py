from mem0 import Memory
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Mem0 Configuration for Hybrid Vector (Qdrant) + Graph (Neo4j)
config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("QDRANT_URL"),
            "port": 6333,
            "collection_name": "yaara_memory"
        }
    },
    "graph_store": {
    "provider": "neo4j",
    "config": {
        "url": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "llm": { # Explicitly tell Mem0 which model to use for GRAPH extraction
            "provider": "openai",
            "config": {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        }
    }
}
}

# Initialize Memory
memory = Memory.from_config(config)