#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini via Google Generative AI

def main():
    if len(sys.argv) < 2:
        print("Usage: run_agent.py 'Your prompt here'")
        sys.exit(1)
    prompt = sys.argv[1]
    
    # Initialize the Gemini model.
    # Replace YOUR_GEMINI_API_KEY with your actual Gemini API key
    llm = ChatGoogleGenerativeAI(
        model_name="gemini-2.0-flash", 
        google_api_key="AIzaSyCCCoVrr42NNT9w0abgabwTUSiuR5qAqK0"
    )
    
    controller = CustomController()
    
    agent = CustomAgent(
        task=prompt,
        llm=llm,
        controller=controller,
        use_vision=False  # Change to True if vision is needed
    )
    
    history = agent.run(max_steps=10)
    print(json.dumps(history.dict(), indent=2))

if __name__ == "__main__":
    main()
