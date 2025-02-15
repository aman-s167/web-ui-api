from flask import Flask, request, jsonify
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from deep_research import deep_research
from src.utils import utils

app = Flask(__name__)

@app.route('/deep_research', methods=['POST'])
def research():
    """Endpoint to trigger deep research via API."""
    try:
        data = request.json
        task = data.get("task", "Default research task")
        max_search_iterations = data.get("max_search_iterations", 3)
        max_query_num = data.get("max_query_num", 3)
        use_own_browser = data.get("use_own_browser", False)

        # Load LLM model (Gemini, OpenAI, etc.)
        llm = utils.get_llm_model(
            provider="gemini",
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            temperature=1.0,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )

        # Run deep research in an async loop inside Flask
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        report_content, report_file_path = loop.run_until_complete(
            deep_research(
                task=task,
                llm=llm,
                max_search_iterations=max_search_iterations,
                max_query_num=max_query_num,
                use_own_browser=use_own_browser
            )
        )

        response = {
            "task": task,
            "report": report_content,
            "report_path": report_file_path
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7788, debug=True)
