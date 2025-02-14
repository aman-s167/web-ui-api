#!/usr/bin/env python3
import sys, os, asyncio, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini via Google Generative AI

async def main():
    if len(sys.argv) < 2:
        print("Usage: run_agent.py 'Your prompt here'")
        sys.exit(1)
    prompt = sys.argv[1]
    
    # Initialize the Gemini model.
    # Replace YOUR_GEMINI_API_KEY with your actual Gemini API key.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key="AIzaSyCCCoVrr42NNT9w0abgabwTUSiuR5qAqK0"
    )
    
    controller = CustomController()
    
    agent = CustomAgent(
        task=prompt,
        llm=llm,
        controller=controller,
        use_vision=False  # Adjust if you need vision
    )
    
    history = await agent.run(max_steps=10)
    print(json.dumps(history.model_dump(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
