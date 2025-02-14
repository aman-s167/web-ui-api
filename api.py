import os
import asyncio
import json
from flask import Flask, request, jsonify

# Hypothetical imports:
# If using Gemini, import GeminiLLM from your custom module.
# Otherwise, import ChatOpenAI for OpenAI.
LLM_TYPE = os.getenv("LLM_TYPE", "openai").lower()

if LLM_TYPE == "gemini":
    # Ensure you have a Gemini adapter module. Replace with the actual implementation.
    from my_gemini_module import GeminiLLM
    # Use the provided Gemini API key and model.
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Optionally, allow setting the Gemini model via an environment variable.
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1")  # Default Gemini model (adjust as needed)
    llm_class = GeminiLLM
    llm_kwargs = {"api_key": gemini_api_key, "model_name": gemini_model}
else:
    # Default to OpenAI's ChatOpenAI adapter
    from langchain_openai import ChatOpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Optionally, allow setting the OpenAI model via an environment variable.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    llm_class = ChatOpenAI
    llm_kwargs = {"api_key": openai_api_key, "model_name": openai_model}

# Instantiate the LLM
llm = llm_class(**llm_kwargs)

# Continue with your agent code...
from src.agent.custom_agent import CustomAgent

app = Flask(__name__)

async def run_agent(prompt: str):
    agent = CustomAgent(task=prompt, llm=llm)
    history = await agent.run(max_steps=10)
    return {"history": str(history)}

async def run_deep_research(prompt: str):
    agent = CustomAgent(task=prompt, llm=llm)
    history = await agent.run(max_steps=10)
    return {"history": str(history)}

@app.route('/api/agent', methods=['POST'])
def api_agent():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    prompt = data['prompt']
    try:
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
    app.run(host='0.0.0.0', port=5000)
