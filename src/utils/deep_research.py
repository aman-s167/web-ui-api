import pdb

from dotenv import load_dotenv

load_dotenv()
import asyncio
import os
import sys
import logging
import json
import re
import time
import random
from pprint import pprint
from uuid import uuid4
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from browser_use.agent.service import Agent
from browser_use.browser.browser import BrowserConfig, Browser
from langchain.schema import SystemMessage, HumanMessage
from json_repair import repair_json
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.controller.custom_controller import CustomController
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import BrowserContextConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
import google.api_core.exceptions

logger = logging.getLogger(__name__)

# API Key Rotation
import itertools
api_keys = itertools.cycle(os.getenv("GOOGLE_API_KEYS", "").split(","))

def get_api_key():
    """Rotate through multiple API keys to avoid rate limits."""
    return next(api_keys).strip()

def invoke_with_retry(messages, retries=3):
    """Invoke the LLM with API key rotation and retry logic."""
    for attempt in range(retries):
        try:
            # Attempt to process the AI query message response
            llm = utils.get_llm_model(
                provider="google",
                model_name="gemini-2.0-flash-thinking-exp-01-21",
                temperature=1.0,
                api_key=get_api_key()
            )
            return llm.invoke(messages)
        except google.api_core.exceptions.ResourceExhausted:
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)  # Exponential backoff (5s, 10s, 15s)
                logging.warning(f"🔄 Rate limit hit (429). Retrying with new key in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("❌ Max retries reached for 429 error.")
                return {"error": "API rate limit exceeded. Please wait and try again later."}

async def deep_research(task, llm, agent_state=None, **kwargs):
    task_id = str(uuid4())
    save_dir = kwargs.get("save_dir", os.path.join(f"./tmp/deep_research/{task_id}"))
    logger.info(f"Save Deep Research at: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    max_query_num = kwargs.get("max_query_num", 3)
    max_search_iterations = kwargs.get("max_search_iterations", 10)
    use_own_browser = kwargs.get("use_own_browser", False)
    extra_chromium_args = []

    if use_own_browser:
        max_query_num = 1
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path == "":
            chrome_path = None
        chrome_user_data = os.getenv("CHROME_USER_DATA", None)
        if chrome_user_data:
            extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=kwargs.get("headless", False),
                disable_security=kwargs.get("disable_security", True),
                chrome_instance_path=chrome_path,
                extra_chromium_args=extra_chromium_args,
            )
        )
        browser_context = await browser.new_context()
    else:
        browser = None
        browser_context = None

    controller = CustomController()
    search_iteration = 0
    history_query = []
    history_infos = []

    try:
        while search_iteration < max_search_iterations:
            search_iteration += 1
            logger.info(f"Start {search_iteration}th Search...")

            query_prompt = f"""
            User Instruction:{task} 
            Previous Queries: {json.dumps(history_query)} 
            Previous Search Results: {json.dumps(history_infos)}
            """
            
            search_messages = [
                SystemMessage(content="Process the following task:"),
                HumanMessage(content=query_prompt)
            ]
            ai_query_msg = invoke_with_retry(search_messages)
            try:
                ai_query_content = json.loads(repair_json(ai_query_msg.content))
                if not isinstance(ai_query_content, dict) or "queries" not in ai_query_content:
                    raise ValueError("Response is not in the expected JSON format with a 'queries' key.")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"JSON decoding error: {e}")
                return {"error": "Invalid response from LLM. Please try again."}, None

            query_tasks = list(set(ai_query_content["queries"]))[:max_query_num]
            if not query_tasks:
                break

            history_query.extend(query_tasks)

            agents = [CustomAgent(
                task=query,
                llm=llm,
                browser=browser,
                browser_context=browser_context,
                controller=controller,
                agent_state=agent_state
            ) for query in query_tasks]

            query_results = await asyncio.gather(
                *[agent.run(max_steps=kwargs.get("max_steps", 10)) for agent in agents]
            )

            for i, query_result in enumerate(query_results):
                if query_result:
                    history_infos.append(query_result.final_result())

        logger.info("\nFinish Searching, Start Generating Report...")
        return history_infos, None  # Ensure two return values
    except Exception as e:
        logger.error(f"Deep research Error: {e}")
        return {"error": str(e)}, None  # Ensure two return values
    finally:
        if browser:
            await browser.close()
        if browser_context:
            await browser_context.close()
        logger.info("Browser closed.")

async def generate_final_report(task, history_infos, save_dir, llm, error_msg=None):
    """Generate report from collected information with error handling"""
    try:
        logger.info("\nAttempting to generate final report from collected data...")
        
        writer_system_prompt = """
        You are a **Deep Researcher** and a professional report writer tasked with creating polished, high-quality reports that fully meet the user's needs, based on the user's instructions and the relevant information provided. You will write the report using Markdown format, ensuring it is both informative and visually appealing.

**Specific Instructions:**

*   **Structure for Impact:** The report must have a clear, logical, and impactful structure. Begin with a compelling introduction that immediately grabs the reader's attention. Develop well-structured body paragraphs that flow smoothly and logically, and conclude with a concise and memorable conclusion that summarizes key takeaways and leaves a lasting impression.
*   **Engaging and Vivid Language:** Employ precise, vivid, and descriptive language to make the report captivating and enjoyable to read. Use stylistic techniques to enhance engagement. Tailor your tone, vocabulary, and writing style to perfectly suit the subject matter and the intended audience to maximize impact and readability.
*   **Accuracy, Credibility, and Citations:** Ensure that all information presented is meticulously accurate, rigorously truthful, and robustly supported by the available data. **Cite sources exclusively using bracketed sequential numbers within the text (e.g., [1], [2], etc.). If no references are used, omit citations entirely.** These numbers must correspond to a numbered list of references at the end of the report.
*   **Publication-Ready Formatting:** Adhere strictly to Markdown formatting for excellent readability and a clean, highly professional visual appearance. Pay close attention to formatting details like headings, lists, emphasis, and spacing to optimize the visual presentation and reader experience. The report should be ready for immediate publication upon completion, requiring minimal to no further editing for style or format.
*   **Conciseness and Clarity (Unless Specified Otherwise):** When the user does not provide a specific length, prioritize concise and to-the-point writing, maximizing information density while maintaining clarity.
*   **Data-Driven Comparisons with Tables:**  **When appropriate and beneficial for enhancing clarity and impact, present data comparisons in well-structured Markdown tables. This is especially encouraged when dealing with numerical data or when a visual comparison can significantly improve the reader's understanding.**
*   **Length Adherence:** When the user specifies a length constraint, meticulously stay within reasonable bounds of that specification, ensuring the content is appropriately scaled without sacrificing quality or completeness.
*   **Comprehensive Instruction Following:** Pay meticulous attention to all details and nuances provided in the user instructions. Strive to fulfill every aspect of the user's request with the highest degree of accuracy and attention to detail, creating a report that not only meets but exceeds expectations for quality and professionalism.
*   **Reference List Formatting:** The reference list at the end must be formatted as follows:  
    `[1] Title (URL, if available)`
    **Each reference must be separated by a blank line to ensure proper spacing.** For example:

    ```
    [1] Title 1 (URL1, if available)

    [2] Title 2 (URL2, if available)
    ```
    **Furthermore, ensure that the reference list is free of duplicates. Each unique source should be listed only once, regardless of how many times it is cited in the text.**
*   **ABSOLUTE FINAL OUTPUT RESTRICTION:**  **Your output must contain ONLY the finished, publication-ready Markdown report. Do not include ANY extraneous text, phrases, preambles, meta-commentary, or markdown code indicators (e.g., "```markdown```"). The report should begin directly with the title and introductory paragraph, and end directly after the conclusion and the reference list (if applicable).**  **Your response will be deemed a failure if this instruction is not followed precisely.**
        
**Inputs:**

1. **User Instruction:** The original instruction given by the user. This helps you determine what kind of information will be useful and how to structure your thinking.
2. **Search Information:** Information gathered from the search queries.
        """

        history_infos_ = json.dumps(history_infos, indent=4)
        record_json_path = os.path.join(save_dir, "record_infos.json")
        logger.info(f"save All recorded information at {record_json_path}")
        with open(record_json_path, "w") as fw:
            json.dump(history_infos, fw, indent=4)
        report_prompt = f"User Instruction:{task} \n Search Information:\n {history_infos_}"
        report_messages = [
            SystemMessage(content=writer_system_prompt),
            HumanMessage(content=report_prompt)
        ]  # New context for report generation
        ai_report_msg = llm.invoke(report_messages)
        if hasattr(ai_report_msg, "reasoning_content"):
            logger.info("🤯 Start Report Deep Thinking: ")
            logger.info(ai_report_msg.reasoning_content)
            logger.info("🤯 End Report Deep Thinking")
        report_content = ai_report_msg.content
        report_content = re.sub(r"^```\s*markdown\s*|^\s*```|```\s*$", "", report_content, flags=re.MULTILINE)
        report_content = report_content.strip()

        # Add error notification to the report if needed
        if error_msg:
            report_content = f"## ⚠️ Research Incomplete - Partial Results\n" \
                             f"**The research process was interrupted by an error:** {error_msg}\n\n" \
                             f"{report_content}"
            
        report_file_path = os.path.join(save_dir, "final_report.md")
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Save Report at: {report_file_path}")
        return report_content, report_file_path

    except Exception as report_error:
        logger.error(f"Failed to generate partial report: {report_error}")
        return f"Error generating report: {str(report_error)}", None
