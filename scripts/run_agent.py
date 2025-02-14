#!/usr/bin/env python3
import sys
import json
from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from langchain_core.language_models.chat_models import BaseChatModel
# Import your LLM of choice. For example, from your utils or llm file:
from src.llm import DeepSeekR1ChatOpenAI  

def main():
    if len(sys.argv) < 2:
        print("Usage: run_agent.py 'Your prompt here'")
        sys.exit(1)
    prompt = sys.argv[1]
    # Initialize your LLM – adjust parameters as needed.
    llm = DeepSeekR1ChatOpenAI(model="deepseek-chat", api_key="YOUR_API_KEY")
    # Initialize controller; if you have custom settings, adjust here.
    controller = CustomController()
    # Instantiate your agent.
    agent = CustomAgent(
        task=prompt,
        llm=llm,
        controller=controller,
        use_vision=False,  # or True if needed
        # additional parameters if needed...
    )
    # Run the agent – for example, with a maximum number of steps.
    history = agent.run(max_steps=10)
    # Output the final history as JSON.
    print(json.dumps(history.dict(), indent=2))

if __name__ == "__main__":
    main()
