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
        # The prompt now includes an explicit JSON example with all required keys.
        text = r"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in exactly the following format:
{
  "current_state": {
    "prev_action_evaluation": "<a string evaluating the previous action, e.g., 'Success' or 'Failed' with explanation>",
    "important_contents": "<a string containing any important content from the page, or an empty string if none>",
    "task_progress": "<a string summarizing completed actions (e.g., '1. Filled username; 2. Filled password; 3. Clicked button')>",
    "future_plans": "<a string outlining what remains to be done>",
    "thought": "<a string containing your internal reasoning about the next steps>",
    "summary": "<a string summarizing the overall process so far>"
  },
  "action": [
    { "action_name": { "param1": "value1", "param2": "value2" } },
    { "action_name": { "param1": "value1" } }
    // ... include up to the maximum number of actions allowed.
  ]
}
Ensure that:
  - All keys inside "current_state" are present and their values are strings (if no content, return an empty string).
  - The "action" field is a list of valid action objects.
    
2. ACTIONS: You can specify multiple actions to be executed in sequence. Use only valid actions as defined in the available functions.

3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list.
   - Each interactive element has a unique numeric index (e.g., "33[:]<button>Submit</button>").
   - Elements marked with "_[:]" are non-interactive (context only).

4. NAVIGATION & ERROR HANDLING:
   - If no suitable elements exist, consider alternative functions.
   - Handle popups/cookies appropriately.
   - Use scrolling if needed to locate elements.

5. TASK COMPLETION:
   - If all user requirements are fulfilled, include a final "Done" action.
   - Do not add extra commentary; only output the JSON as specified.

6. VISUAL CONTEXT:
   - When an image is provided, use it to understand the page layout.
   - Bounding boxes and their labels can help verify element positions.

7. FORM FILLING:
   - When filling an input field, if suggestions appear, select the correct option before proceeding.

8. ACTION SEQUENCING:
   - List actions in the order they should be executed.
   - If the page changes after an action, the sequence may be interrupted.
   - Provide only as many actions as needed until the page is expected to change.
   - Do not include extra information outside the JSON structure.

Please follow this exact format for your output.
        """
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Task: The user's instructions you need to complete.
2. Hints(Optional): Any hints to help you complete the task.
3. Memory: Previously recorded important content (if any).
4. Current URL: The current page URL.
5. Available Tabs: List of open browser tabs.
6. Interactive Elements: Provided as a list in the format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier.
   - element_type: e.g., button, input.
   - element_text: Visible text.
Example:
33[:]<button>Submit Form</button>
_[:] Non-interactive text.
Notes:
- Only elements with numeric indexes are interactive.
- _[:] elements provide context only.
        """

    def get_system_message(self) -> SystemMessage:
        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
1. Analyze the provided webpage elements and structure.
2. Plan a sequence of actions to accomplish the given task.
3. Your final result MUST be a valid JSON following the **RESPONSE FORMAT** exactly as described below. Do not include any extra text or commentary.
{self.input_format()}
{self.important_rules()}
Functions:
{self.default_action_description}
Remember: Your response must be valid JSON matching the specified format."""
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
