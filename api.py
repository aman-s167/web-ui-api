import os
import asyncio
from flask import Flask, request, jsonify

from src.agent.custom_agent import CustomAgent
from langchain_openai import ChatOpenAI

# Import the GeminiLLM wrapper from our newly created module.
try:
    from src.llm.gemini_llm import GeminiLLM
except ImportError:
    GeminiLLM = None

def get_llm_instance():
    llm_type = os.getenv("LLM_TYPE", "openai").lower()
    if llm_type == "gemini":
        if GeminiLLM is None:
            raise ImportError("GeminiLLM module is not installed or imported properly.")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Default to gemini-2.0-flash (adjust as needed)
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when LLM_TYPE is 'gemini'.")
        return GeminiLLM(api_key=gemini_api_key, model_name=gemini_model)
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI LLM.")
        return ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")

app = Flask(__name__)

@app.route('/api/agent', methods=['POST'])
def api_agent():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    prompt = data['prompt']
    try:
        llm = get_llm_instance()
        agent = CustomAgent(task=prompt, llm=llm)
        history = asyncio.run(agent.run(max_steps=10))
        return jsonify({'result': {"history": str(history)}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/deep-research', methods=['POST'])
def api_deep_research():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    prompt = data['prompt']
    try:
        llm = get_llm_instance()
        agent = CustomAgent(task=prompt, llm=llm)
        history = asyncio.run(agent.run(max_steps=10))
        return jsonify({'result': {"history": str(history)}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
