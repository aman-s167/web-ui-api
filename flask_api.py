import os
import json
import asyncio
import logging
import redis
from flask import Flask, request, jsonify
from uuid import uuid4
import utils  # Assuming utils has functions for getting API keys and LLM models

# Initialize Flask app
app = Flask(__name__)

# Configure Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

# Redis Helper Functions
def get_cache_key(task):
    """Generate a unique cache key for the research task."""
    return f"research_cache:{hash(task)}"

def cache_results(task, data):
    """Save research results in Redis."""
    cache_key = get_cache_key(task)
    redis_client.set(cache_key, json.dumps(data), ex=3600)  # Cache for 1 hour

def load_cached_results(task):
    """Retrieve cached research results if available."""
    cache_key = get_cache_key(task)
    cached_data = redis_client.get(cache_key)
    return json.loads(cached_data) if cached_data else None

@app.route('/api/research', methods=['POST'])
def handle_research():
    data = request.get_json()
    
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing required field: task'}), 400

    task = data['task']
    cached_results = load_cached_results(task)  # ✅ Check Redis Cache

    if cached_results:
        return jsonify({'status': 'success', 'cached_report': cached_results})

    try:
        llm = utils.get_llm_model(
            provider="google",
            model_name="gemini-2.0-flash-exp",
            temperature=1.0,
            api_key=utils.get_api_key()
        )

        from deep_research import deep_research  # Importing the function dynamically
        report_content, _ = asyncio.run(deep_research(task, llm))

        # Store results in Redis
        cache_results(task, report_content)

        return jsonify({'status': 'success', 'report': report_content})

    except Exception as e:
        logging.error(f"Error processing research: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)
