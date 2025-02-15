import os
import asyncio
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import deep_research function from the Browser Use Web UI
from deep_research import deep_research
from src.utils import utils

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/api/research', methods=['POST'])
def handle_research():
    data = request.get_json()
    
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing required field: task'}), 400
    
    task = data['task']
    max_search_iterations = data.get('max_search_iterations', 5)
    max_query_num = data.get('max_query_num', 3)
    use_own_browser = data.get('use_own_browser', False)
    
    try:
        llm = utils.get_llm_model(
            provider="gemini",
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            temperature=1.0,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        
        # Run deep_research asynchronously
        report_content, _ = asyncio.run(deep_research(
            task=task, 
            llm=llm, 
            agent_state=None, 
            max_search_iterations=max_search_iterations, 
            max_query_num=max_query_num, 
            use_own_browser=use_own_browser
        ))
        
        return jsonify({'status': 'success', 'report': report_content})
    except Exception as e:
        logging.error(f"Error processing research: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
