from __future__ import annotations

import logging
from typing import List, Optional, Type

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, AgentStepInfo, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage
)
from langchain_openai import ChatOpenAI
from ..utils.llm import DeepSeekR1ChatOpenAI
from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)


class CustomMessageManager(MessageManager):
    def __init__(
        self,
        llm: BaseChatModel,
        task: str,
        action_descriptions: str,
        system_prompt_class: Type[SystemPrompt],
        agent_prompt_class: Type[AgentMessagePrompt],
        max_input_tokens: int = 128000,
        estimated_characters_per_token: int = 3,
        image_tokens: int = 800,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        message_context: Optional[str] = None,
    ):
        super().__init__(
            llm=llm,
            task=task,
            action_descriptions=action_descriptions,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            estimated_characters_per_token=estimated_characters_per_token,
            image_tokens=image_tokens,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            message_context=message_context,
        )
        self.agent_prompt_class = agent_prompt_class
        # Custom: Initialize history with system prompt and optional context.
        self.history = MessageHistory()
        self._add_message_with_tokens(self.system_prompt)

        if self.message_context:
            context_message = HumanMessage(content=self.message_context)
            self._add_message_with_tokens(context_message)

    def cut_messages(self):
        """Trim the message history to keep it under the maximum input tokens."""
        diff = self.history.total_tokens - self.max_input_tokens
        min_message_len = 2 if self.message_context is not None else 1

        while diff > 0 and len(self.history.messages) > min_message_len:
            self.history.remove_message(min_message_len)  # always remove the oldest message
            diff = self.history.total_tokens - self.max_input_tokens

    def add_state_message(
        self,
        state: BrowserState,
        actions: Optional[List[ActionModel]] = None,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """Add the browser state as a human message. Note that we do not pass 'actions' since our prompt class doesn't expect it."""
        state_message = self.agent_prompt_class(
            state=state,
            result=result,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message()
        self._add_message_with_tokens(state_message)

    def _count_text_tokens(self, text: str) -> int:
        if isinstance(self.llm, (ChatOpenAI, ChatAnthropic, DeepSeekR1ChatOpenAI)):
            try:
                tokens = self.llm.get_num_tokens(text)
            except Exception:
                tokens = len(text) // self.estimated_characters_per_token  # rough estimate if tokenizer is unavailable
        else:
            tokens = len(text) // self.estimated_characters_per_token  # rough estimate
        return tokens

    def _remove_state_message_by_index(self, remove_ind=-1) -> None:
        """Remove the last state message from history based on the provided index."""
        i = len(self.history.messages) - 1
        remove_cnt = 0
        while i >= 0:
            if isinstance(self.history.messages[i].message, HumanMessage):
                remove_cnt += 1
            if remove_cnt == abs(remove_ind):
                self.history.remove_message(i)
                break
            i -= 1
