import os
import json
import asyncio
import hashlib
import redis
from uuid import uuid4
from logger import logger
from langchain.schema import SystemMessage, HumanMessage
from utils import invoke_with_retry, repair_json

# Configure Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

# Redis Helper Functions
def get_cache_key(task):
    """Generate a unique cache key for the research task."""
    return f"research_cache:{hashlib.sha256(task.encode()).hexdigest()}"

def cache_results(task, data):
    """Save research results in Redis."""
    cache_key = get_cache_key(task)
    redis_client.set(cache_key, json.dumps(data), ex=3600)  # Cache for 1 hour

def load_cached_results(task):
    """Retrieve cached research results if available."""
    cache_key = get_cache_key(task)
    cached_data = redis_client.get(cache_key)
    return json.loads(cached_data) if cached_data else None

async def deep_research(task, llm, agent_state=None, **kwargs):
    """Perform deep research while caching results to prevent redundant API calls."""
    cached_data = load_cached_results(task)
    if cached_data:
        logger.info("✅ Using cached research results.")
        return cached_data, None  # Return cached results instead of fetching again

    # Proceed with normal research if no cached data exists
    task_id = str(uuid4())
    save_dir = kwargs.get("save_dir", os.path.join(f"./tmp/deep_research/{task_id}"))
    logger.info(f"Save Deep Research at: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    max_query_num = kwargs.get("max_query_num", 3)
    max_search_iterations = kwargs.get("max_search_iterations", 10)

    history_query = []
    history_infos = []

    try:
        for search_iteration in range(max_search_iterations):
            logger.info(f"🔍 Starting iteration {search_iteration+1}")

            query_prompt = f"""
User Instruction: {task}
Provide a valid JSON object with "queries" key.
Previous Queries: {json.dumps(history_query)}
Previous Search Results: {json.dumps(history_infos)}
            """

            search_messages = [
                SystemMessage(content="Generate JSON with 'queries'"),
                HumanMessage(content=query_prompt)
            ]
            ai_query_msg = invoke_with_retry(search_messages)

            try:
                ai_query_content = json.loads(repair_json(ai_query_msg.content))
                if not isinstance(ai_query_content, dict) or "queries" not in ai_query_content:
                    raise ValueError("Invalid JSON format.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON decoding error: {e}")
                return {"error": "Invalid response from LLM."}, None

            query_tasks = list(set(ai_query_content["queries"]))[:max_query_num]
            if not query_tasks:
                break

            history_query.extend(query_tasks)

            # Run all queries asynchronously
            query_results = await asyncio.gather(
                *[CustomAgent(task=q, llm=llm, agent_state=agent_state).run() for q in query_tasks]
            )

            for result in query_results:
                if result:
                    history_infos.append(result.final_result())

        logger.info("✅ Research complete. Saving results...")
        cache_results(task, history_infos)  # ✅ Save results to Redis
        return history_infos, None

    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        return {"error": str(e)}, None
