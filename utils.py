import os
from server import settings


def call_llm(messages, model, temperature=0.0):
    """Call LLM API"""
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatZhipuAI
    
    # Get model configuration
    model_config = settings.api_model_settings.MODELS.get(model)
    platform_name = model_config.platform_name if model_config else "openai"
    
    # Get platform configuration
    platform_config = None
    for platform in settings.api_model_settings.MODEL_PLATFORMS:
        if platform.platform_name == platform_name:
            platform_config = platform
            break
    
    if platform_name == "openai":
        llm = ChatOpenAI(
            model=model,
            api_key=platform_config.api_key if platform_config else None,
            base_url=platform_config.api_llm_base_url if platform_config else None,
            temperature=temperature
        )
    elif platform_name == "zhipuai":
        llm = ChatZhipuAI(
            model=model,
            api_key=platform_config.api_key if platform_config else None,
            base_url=platform_config.api_llm_base_url if platform_config else None,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported platform: {platform_name}")
    
    response = llm.invoke(messages)
    return response.content

