import os
import sys
import json
import logging
from typing import Optional

from langchain.schema import HumanMessage
from src.agent.custom_agent import CustomAgent
from src.message_manager.custom_message_manager import CustomMessageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_agent.py '<your_command>'")
        sys.exit(1)
    
    user_prompt = sys.argv[1]
    
    logger.info("\U0001F680 Starting task: %s", user_prompt)
    
    # Initialize message manager
    message_manager = CustomMessageManager()
    
    # Add user-provided input as a primary instruction
    message_manager._add_message_with_tokens(HumanMessage(content=f"USER INSTRUCTION: {user_prompt}"))
    
    # Initialize the agent
    agent = CustomAgent(
        message_manager=message_manager,
        browser_env_name=os.getenv("BROWSER_ENV", "default"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", 10))
    )
    
    # Run the agent
    result = agent.run()
    
    # Output the result
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    main()
