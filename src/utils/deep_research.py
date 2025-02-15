import os
import asyncio
import json
import time
import logging
import google.api_core.exceptions
from dotenv import load_dotenv
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from browser_use.browser.browser import BrowserConfig, Browser
from browser_use.browser.context import BrowserContextConfig

# Load environment variables
load_dotenv()

CACHE_FILE = "query_cache.json"

logger = logging.getLogger(__name__)


def load_cache():
    """Load cached results from a file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save results to cache."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


cache = load_cache()


def get_api_key():
    """Rotate between multiple API keys to avoid rate limits."""
    keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
    return keys[0].strip() if keys else os.getenv("GOOGLE_API_KEY", "")


def invoke_with_retry(llm, messages, retries=3):
    """Invoke the LLM with retry logic for handling rate limits."""
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)  # Exponential backoff (5s, 10s, 15s)
                print(f"🔄 Rate limit hit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception("Max retries reached for 429 error.")


async def deep_research(task, llm, agent_state=None, **kwargs):
    max_search_iterations = kwargs.get("max_search_iterations", 5)
    max_query_num = kwargs.get("max_query_num", 3)

    history_query = []
    history_infos = []

    for search_iteration in range(max_search_iterations):
        logger.info(f"Start {search_iteration + 1}th Search...")
        query_prompt = f"User Instruction: {task}\nPrevious Queries: {json.dumps(history_query)}"

        ai_query_msg = invoke_with_retry(llm, [query_prompt])
        ai_query_content = json.loads(ai_query_msg.content)

        query_tasks = list(set(ai_query_content["queries"]))[:max_query_num]  # Remove duplicates
        history_query.extend(query_tasks)

        if not query_tasks:
            break

        for query in query_tasks:
            if query in cache:
                print(f"✅ Using cached result for: {query}")
                history_infos.append(cache[query])
                continue

            agent = CustomAgent(task=query, llm=llm)
            agent_result = await agent.run()
            query_results = agent_result.final_result()

            cache[query] = query_results
            save_cache(cache)

            history_infos.append(query_results)

    logger.info("\nFinish Searching, Start Generating Report...")
    return history_infos
