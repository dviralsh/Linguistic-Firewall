from core.tools import fetch_financial_data, fetch_news_headlines, calculate_percentage_change, malicious_tool_instance, system_diagnostic_upload

TOOL_REGISTRY = {
    "fetch_financial_data": fetch_financial_data,
    "fetch_news_headlines": fetch_news_headlines,
    "calculate_percentage_change": calculate_percentage_change,
    "system_diagnostic_upload": system_diagnostic_upload
}

def get_tool_function(tool_name: str):
    return TOOL_REGISTRY.get(tool_name)
