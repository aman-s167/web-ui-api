#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from src.llm import DeepSeekR1ChatOpenAI

def main():
    if len(sys.argv) < 2:
        print("Usage: run_agent.py 'Your prompt here'")
        sys.exit(1)
    prompt = sys.argv[1]
    # Initialize your LLM – adjust parameters as needed.
    llm = DeepSeekR1ChatOpenAI(model="deepseek-chat", api_key="YOUR_API_KEY")
    # Initialize controller.
    controller = CustomController()
    # Instantiate your agent.
    agent = CustomAgent(
        task=prompt,
        llm=llm,
        controller=controller,
        use_vision=False  # or True if needed
    )
    # Run the agent – adjust the max steps as needed.
    history = agent.run(max_steps=10)
    # Output the final history as JSON.
    print(json.dumps(history.dict(), indent=2))

if __name__ == "__main__":
    main()
