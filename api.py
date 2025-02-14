# api.py
import json
import asyncio
from flask import Flask, request, jsonify

# IMPORTANT: Adjust the following import if your CustomAgent is in a different location.
# Here we assume that CustomAgent is defined in a file inside the project.
from src.agent.custom_agent import CustomAgent
from langchain_openai import ChatOpenAI

# Helper function to run the agent (agent endpoint)
async def run_agent(prompt: str):
    # Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
    llm = ChatOpenAI(api_key='YOUR_OPENAI_API_KEY', model_name='gpt-3.5-turbo')
    # Create an instance of the agent using the prompt as the task.
    agent = CustomAgent(task=prompt, llm=llm)
    # Run the agent for a maximum of 10 steps (adjust if needed)
    history = await agent.run(max_steps=10)
    # Return the history as a string (for simplicity)
    return {"history": str(history)}

# Helper function for deep research (deep research endpoint)
async def run_deep_research(prompt: str):
    llm = ChatOpenAI(api_key='YOUR_OPENAI_API_KEY', model_name='gpt-3.5-turbo')
    agent = CustomAgent(task=prompt, llm=llm)
    history = await agent.run(max_steps=10)
    return {"history": str(history)}

# Create the Flask app
app = Flask(__name__)

@app.route('/api/agent', methods=['POST'])
def api_agent():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    prompt = data['prompt']
    try:
        # Run the asynchronous function and get the result.
        result = asyncio.run(run_agent(prompt))
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/deep-research', methods=['POST'])
def api_deep_research():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    prompt = data['prompt']
    try:
        result = asyncio.run(run_deep_research(prompt))
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # The API will listen on port 5000.
    app.run(host='0.0.0.0', port=5000)
