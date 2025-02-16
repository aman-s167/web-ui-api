import os
import asyncio
import logging
import time
import google.api_core.exceptions
import random
import redis
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from src.utils.deep_research import deep_research
from src.utils import utils

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ✅ Load Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6381/0")
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

def get_api_key():
    """Rotate between multiple API keys to avoid rate limits."""
    keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
    return random.choice(keys).strip() if keys else os.getenv("GOOGLE_API_KEY", "")

@app.route('/api/research', methods=['POST'])
def handle_research():
    data = request.get_json()
    
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing required field: task'}), 400
    
    task = data['task']
    max_search_iterations = data.get('max_search_iterations', 5)
    max_query_num = data.get('max_query_num', 3)
    use_own_browser = data.get('use_own_browser', False)

    # ✅ Check if result is cached in Redis
    cache_key = f"research:{task}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return jsonify({'status': 'success', 'cached': True, 'report': cached_result})

    try:
        llm = utils.get_llm_model(
            provider="google",
            model_name="gemini-2.0-flash-exp",
            temperature=1.0,
            api_key=get_api_key()
        )
        
        retries = 3  # Number of retries for rate limit errors
        for attempt in range(retries):
            try:
                report_content, _ = asyncio.run(deep_research(
                    task=task, 
                    llm=llm, 
                    agent_state=None, 
                    max_search_iterations=max_search_iterations, 
                    max_query_num=max_query_num, 
                    use_own_browser=use_own_browser
                ))

                # ✅ Store result in Redis cache (expires in 24 hours)
                redis_client.setex(cache_key, 86400, report_content)

                return jsonify({'status': 'success', 'cached': False, 'report': report_content})
            
            except google.api_core.exceptions.ResourceExhausted as e:
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff (5s, 10s, 15s)
                    logging.warning(f"Rate limit hit (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries reached for 429 error.")
                    return jsonify({'status': 'error', 'message': 'API rate limit exceeded. Please wait and try again later.'}), 429
    
    except Exception as e:
        logging.error(f"Error processing research: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
