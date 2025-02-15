import os
import sys
# Ensure the project root is in sys.path so that modules inside src can be imported.
sys.path.insert(0, os.getcwd())
print("sys.path:", sys.path)  # For debugging

import asyncio
import logging
import random
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.utils.agent_state import AgentState  # AgentState is now in src/utils/agent_state.py
from browser_use.browser.browser import Browser, BrowserConfig
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import BrowserContextConfig
from browser_use.browser.context import BrowserContextConfig as BU_BrowserContextConfig, BrowserContextWindowSize

# Load environment variables and initialize Flask
load_dotenv()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def get_api_key():
    keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
    return random.choice(keys).strip() if keys and keys[0] else os.getenv("GOOGLE_API_KEY", "")

@app.route('/api/agent', methods=['POST'])
def handle_agent():
    data = request.get_json()
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing required field: task'}), 400

    task = data['task']
    max_steps = data.get('max_steps', 10)
    use_own_browser = data.get('use_own_browser', False)

    try:
        # Initialize the LLM model (example using Google Gemini)
        llm = utils.get_llm_model(
            provider="google",
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            temperature=1.0,
            api_key=get_api_key()
        )

        # Set up controller and agent state
        controller = CustomController()
        agent_state = AgentState()

        # Set up browser or use default context if not using a dedicated browser instance
        browser = None
        browser_context = None
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            extra_chromium_args = []
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")
            browser = CustomBrowser(
                config=BrowserConfig(
                    headless=True,  # set to False if you need to see the browser
                    disable_security=True,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args
                )
            )
            browser_context = asyncio.run(browser.new_context(
                config=BrowserContextConfig(
                    trace_path="./tmp/traces",
                    save_recording_path="./tmp/record_videos",
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=1920, height=1080)
                )
            ))

        # Instantiate and run the CustomAgent
        agent = CustomAgent(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            agent_state=agent_state
        )

        result = asyncio.run(agent.run(max_steps=max_steps))
        final_report = result.final_result()

        # Close browser context and browser if created
        if browser_context:
            asyncio.run(browser_context.close())
        if browser:
            asyncio.run(browser.close())

        return jsonify({'status': 'success', 'report': final_report})
    
    except Exception as e:
        logging.error(f"Error processing agent: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Disable the reloader to avoid asyncio event loop conflicts.
    app.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)
