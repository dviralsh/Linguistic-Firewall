from autogen_ext.models.openai import OpenAIChatCompletionClient
from config.settings import get_api_config_for_model


class ClientFactory:
    @staticmethod
    def get_client(
        model_name: str,
        base_url: str = None,
        api_key: str = None,
        **kwargs,
    ):
        if base_url is None or api_key is None:
            config_base, config_key = get_api_config_for_model(model_name)
            base_url = base_url or config_base
            api_key = api_key or config_key

        client_kwargs = {
            "model": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "model_info": {
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
            },
            "temperature": 0.1,
            "max_tokens": 256,
            "timeout": 120,
        }
        client_kwargs.update(kwargs)

        return OpenAIChatCompletionClient(**client_kwargs)
