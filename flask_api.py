import os
import redis
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6381/0")
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    # ✅ Check if result is already cached
    cache_key = f"search:{query}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return jsonify({"cached": True, "result": cached_result})

    # Perform search (replace with actual logic)
    result = f"Simulated search results for: {query}"

    # ✅ Store result in Redis cache (expires in 24 hours)
    redis_client.setex(cache_key, 86400, result)

    return jsonify({"cached": False, "result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
