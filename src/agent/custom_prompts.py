import pdb
from typing import List, Optional
from datetime import datetime

from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage

from .custom_views import CustomAgentStepInfo


class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        text = r"""
    1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
       {
         "current_state": {
           "prev_action_evaluation": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful as intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why or why not. If you consider it to be 'Failed', reflect on it in your thought.",
           "important_contents": "Output important contents closely related to the user's instruction on the current page. If none, output an empty string.",
           "task_progress": "Summarize the completed steps. List each completed item (e.g., '1. Input username. 2. Input password. 3. Click confirm button') as a string.",
           "future_plans": "Outline the remaining steps required to complete the task (as a string).",
           "thought": "Reflect on what has been done and what needs to be done next. If a previous action failed, include your reflection here.",
           "summary": "Provide a brief description for the next operations based on your thought."
         },
         "action": [
           * actions in sequence; each action must be formatted as: {action_name: action_params} *
         ]
       }
    2. ACTIONS: You can specify multiple actions to be executed in sequence.
       Common sequences include:
       - Form filling: [
           {"input_text": {"index": 1, "text": "username"}},
           {"input_text": {"index": 2, "text": "password"}},
           {"click_element": {"index": 3}}
         ]
       - Navigation: [
           {"go_to_url": {"url": "https://example.com"}},
           {"extract_page_content": {}}
         ]
    3. ELEMENT INTERACTION:
       - Only use element indexes provided.
       - Only elements with numeric indexes are interactive.
       - Elements marked with "_[:]" are for context only.
    4. NAVIGATION & ERROR HANDLING:
       - Use alternative methods if no suitable elements are available.
       - Handle popups/cookies appropriately.
    5. TASK COMPLETION:
       - If all requirements are met, output the Done action to terminate the process.
       - Do not hallucinate actions.
       - Always verify fulfillment by checking the page content.
    6. VISUAL CONTEXT:
       - When an image is provided, use it to understand the page layout and element positions.
    7. FORM FILLING:
       - If suggestions appear under a filled input, select the correct suggestion.
    8. ACTION SEQUENCING:
       - Execute actions in the order they are listed.
       - Only include actions until you expect a page change.
       - Use up to {max_actions_per_step} actions per sequence.
        """
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
    INPUT STRUCTURE:
    1. Task: The user's instruction.
    2. Hints (Optional): Additional hints.
    3. Memory: Important content recorded from previous steps.
    4. Current URL: The current webpage URL.
    5. Available Tabs: List of open browser tabs.
    6. Interactive Elements: A list formatted as:
       index[:]<element_type>element_text</element_type>
       (Example: 33[:]<button>Submit Form</button>; _[:] indicates non-interactive elements.)
        """

    def get_system_message(self) -> SystemMessage:
        AGENT_PROMPT = f"""You are a precise browser automation agent. Your role is to:
    1. Analyze the provided webpage elements.
    2. Plan a sequence of actions to accomplish the task.
    3. Return a valid JSON response as specified, containing your planned actions and state assessment.
    {self.input_format()}
    {self.important_rules()}
    Functions:
    {self.default_action_description}
    Respond strictly in JSON.
        """
        return SystemMessage(content=AGENT_PROMPT)


class CustomAgentMessagePrompt(AgentMessagePrompt):
    def __init__(self, *args, actions: Optional[List[ActionModel]] = None, **kwargs):
        # Remove any 'include_attributes' from kwargs.
        kwargs.pop('include_attributes', None)
        # Initialize with include_attributes explicitly set to an empty list.
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
        if elements_text:
            if self.state.pixels_above:
                elements_text = f"... {self.state.pixels_above} pixels above - scroll to see more...\n{elements_text}"
            else:
                elements_text = f"[Start of page]\n{elements_text}"
            if self.state.pixels_below:
                elements_text = f"{elements_text}\n... {self.state.pixels_below} pixels below - scroll to see more..."
            else:
                elements_text = f"{elements_text}\n[End of page]"
        else:
            elements_text = "empty page"
        state_description = f"""
{step_info_description}
1. Task: {self.step_info.task}.
2. Hints (Optional):
{self.step_info.add_infos}
3. Memory:
{self.step_info.memory}
4. Current URL: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}
        """
        if self.actions and self.result:
            state_description += f"\n **Previous Actions** \nPrevious step: {self.step_info.step_number-1}/{self.step_info.max_steps}\n"
            def flatten_and_stringify(err):
                flat = []
                if isinstance(err, list):
                    for item in err:
                        flat.extend(flatten_and_stringify(item))
                elif isinstance(err, dict):
                    flat.append(str(err))
                else:
                    flat.append(str(err))
                return flat
            for i, res in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i+1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if res.include_in_memory:
                    if res.extracted_content:
                        state_description += f"Result of previous action {i+1}/{len(self.result)}: {res.extracted_content}\n"
                    if res.error:
                        error_list = flatten_and_stringify(res.error)
                        error_str = ", ".join(error_list)[-self.max_error_length:]
                        state_description += f"Error of previous action {i+1}/{len(self.result)}: {error_str}\n"
        if self.state.screenshot:
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_description},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.state.screenshot}"}}
                ]
            )
        return HumanMessage(content=state_description)
