import os
import redis
import time
import requests

# ✅ Load Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

def deep_search(query, max_retries=3, wait_time=10):
    cache_key = f"deep_search:{query}"
    
    # ✅ Check if data is already cached
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print(f"✅ Returning cached data for: {query}")
        return cached_data

    retries = 0
    while retries < max_retries:
        try:
            # Simulate API call (Replace with actual research function)
            response = requests.get(f"https://example.com/search?q={query}")
            response.raise_for_status()
            data = response.text
            
            # ✅ Cache the data to prevent redundant API calls
            redis_client.setex(cache_key, 86400, data)
            return data

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching data: {e}, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1

    print(f"❌ Failed after {max_retries} retries. Returning last cached result if available.")
    return redis_client.get(cache_key)  # Return last cached result if exists

if __name__ == "__main__":
    query = "latest AI research"
    print(deep_search(query))
