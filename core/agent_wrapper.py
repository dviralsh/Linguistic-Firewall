from typing import List, Optional, AsyncGenerator, Any, Sequence
import asyncio
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelCapabilities,
    ModelInfo,
    LLMMessage,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from core.factory import ClientFactory
from core.registry import get_tool_function


class MockModelClient(ChatCompletionClient):
    """
    A lightweight dummy client that returns fixed text and uses 0 RAM.
    """

    def __init__(self, response: str = ""):
        self._response = response

    async def create(self, messages: Sequence[LLMMessage], **kwargs) -> CreateResult:
        # FIXED: Added required fields 'cached' and specific 'usage' keys
        return CreateResult(
            content=self._response,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            cached=False,
        )

    @property
    def model_info(self):
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "mock",
        }

    @property
    def capabilities(self):
        return ModelCapabilities(vision=False, function_calling=True, json_output=False)

    async def create_stream(
        self, messages: Sequence[LLMMessage], **kwargs
    ) -> AsyncGenerator[Any, None]:
        # FIXED: Added required fields here as well
        yield CreateResult(
            content=self._response,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            cached=False,
        )

    def remaining_tokens(self, messages: Sequence[LLMMessage]) -> int:
        return 999999

    async def close(self):
        pass

    def actual_usage(self):
        return 0

    def total_usage(self):
        return 0

    def count_tokens(self, messages):
        return 0


class AgentWrapper:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.model_name = config.get("model", "default")
        self.system_message = config.get("system_message", "")
        self.description = config.get("description", "No description provided.")
        self.tool_names = config.get("tools", [])

        if "model_client" in config:
            self.client = config["model_client"]
        else:
            timeout = config.get("timeout")
            if timeout:
                self.client = ClientFactory.get_client(self.model_name, timeout=timeout)
            else:
                self.client = ClientFactory.get_client(self.model_name)

        self.tools = [
            get_tool_function(t) for t in self.tool_names if get_tool_function(t)
        ]

        self.context_strategy = BufferedChatCompletionContext(buffer_size=10)

        # User requested per-agent configuration for reflection
        extra_kwargs = {}
        if self.name == "Malicious":
            extra_kwargs["reflect_on_tool_use"] = True

        self.agent = AssistantAgent(
            name=self.name,
            description=self.description,
            system_message=self.system_message,
            model_client=self.client,
            tools=self.tools,
            model_context=self.context_strategy,
            **extra_kwargs,
        )

    def get_agent(self) -> AssistantAgent:
        return self.agent

    async def close(self):
        if hasattr(self.client, "close"):
            if asyncio.iscoroutinefunction(self.client.close):
                await self.client.close()
            else:
                self.client.close()
