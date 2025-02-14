import pdb
from typing import List, Optional
from datetime import datetime
import logging

from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage

from .custom_views import CustomAgentStepInfo

logger = logging.getLogger(__name__)

class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = r"""
1. RESPONSE FORMAT: Your final output MUST be a single valid JSON object exactly in the following format with no extra text. Every key under "current_state" must be present. If there is no value for a key, output an empty string (""). The JSON must exactly follow this structure:

{
  "current_state": {
    "prev_action_evaluation": "<string>",
    "important_contents": "<string>",
    "task_progress": "<string>",
    "future_plans": "<string>",
    "thought": "<string>",
    "summary": "<string>"
  },
  "action": [
    { "action_name": { "param1": "value1", "param2": "value2" } },
    // ... additional actions, if any
  ]
}

Important:
  - All keys inside "current_state" are required.
  - If no data exists for a key, output an empty string ("").
  - The "action" field must be an array (which may be empty).
  
2. ACTIONS: You may list multiple actions if needed, but only include valid actions as defined in your functions.
3. ELEMENT INTERACTION: Use only the provided interactive elements with numeric indexes.
4. NAVIGATION & ERROR HANDLING: Follow standard guidelines (handle popups, use scrolling, etc.).
5. TASK COMPLETION: When all requirements are met, include a "Done" action in the list.
6. VISUAL CONTEXT: Use provided visual data only as context. Do not include extra keys.
7. FORM FILLING: Ensure to handle form fields correctly, including selecting suggestions if needed.
8. ACTION SEQUENCING: List actions in the correct order. Only include as many as needed.

Please ensure your response is exactly the JSON object as specified.
        """
        text += f"\n   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Task: The user's instructions.
2. Hints(Optional): Any additional hints.
3. Memory: Previously recorded content (if any).
4. Current URL: The current webpage URL.
5. Available Tabs: A list of open browser tabs.
6. Interactive Elements: Provided as a list in the following format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier.
   - element_type: e.g., button, input.
   - element_text: Visible text.
Example:
33[:]<button>Submit Form</button>
_[:] Non-interactive text.
Notes:
- Only numeric-indexed elements are interactive.
        """

    def get_system_message(self) -> SystemMessage:
        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
1. Analyze the provided webpage elements and structure.
2. Plan a sequence of actions to accomplish the given task.
3. Your final output MUST be a valid JSON object exactly following this schema:

{{
  "current_state": {{
    "prev_action_evaluation": "<string>",
    "important_contents": "<string>",
    "task_progress": "<string>",
    "future_plans": "<string>",
    "thought": "<string>",
    "summary": "<string>"
  }},
  "action": [
    {{ "action_name": {{ "param1": "value1", "param2": "value2" }} }},
    ...
  ]
}}

Do not include any additional keys or text. If no information exists for a field, output an empty string ("").

{self.input_format()}
{self.important_rules()}
Functions:
{self.default_action_description}
Remember: Your response must be exactly the JSON object in the format above."""
        return SystemMessage(content=AGENT_PROMPT)


class CustomAgentMessagePrompt(AgentMessagePrompt):
    def __init__(self, *args, actions: Optional[List[ActionModel]] = None, **kwargs):
        # Remove any 'include_attributes' from kwargs.
        kwargs.pop('include_attributes', None)
        # Call the base initializer with include_attributes explicitly set to an empty list.
        super().__init__(
            state=kwargs.get('state'),
            result=kwargs.get('result'),
            include_attributes=[],
            max_error_length=kwargs.get('max_error_length', 400),
            step_info=kwargs.get('step_info')
        )
        self.actions = actions
        self.include_attributes = []  # Force an empty list

    def get_user_message(self) -> HumanMessage:
        if self.step_info:
            step_info_description = f"Current step: {self.step_info.step_number}/{self.step_info.max_steps}\n"
        else:
            step_info_description = ""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        step_info_description += f"Current date and time: {time_str}"
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0
        if elements_text != "":
            if has_content_above:
                elements_text = f"... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}"
            else:
                elements_text = f"[Start of page]\n{elements_text}"
            if has_content_below:
                elements_text = f"{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ..."
            else:
                elements_text = f"{elements_text}\n[End of page]"
        else:
            elements_text = "empty page"
        state_description = f"""
{step_info_description}
1. Task: {self.step_info.task}.
2. Hints(Optional):
{self.step_info.add_infos}
3. Memory:
{self.step_info.memory}
4. Current url: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}
        """
        if self.actions and self.result:
            state_description += "\n **Previous Actions** \n"
            state_description += f"Previous step: {self.step_info.step_number-1}/{self.step_info.max_steps} \n"
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {str(action.model_dump_json(exclude_unset=True))}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {str(result.extracted_content)}\n"
                    if result.error:
                        error_str = str(result.error)
                        state_description += f"Error of previous action {i + 1}/{len(self.result)}: ...{error_str}\n"
        if self.state.screenshot:
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_description},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.state.screenshot}"}}
                ]
            )
        return HumanMessage(content=state_description)
