import os
import json
import pickle
from server import settings


def load_config():
    """Load configuration from settings"""
    return {
        "openai": {
            "model": settings.api_model_settings.SCENARIO_MODELS.get("extract_entity", "GPT-4o-mini")
        },
        "graph": {
            "community_detection": "leiden"
        },
        "paths": {
            "output_dir": str(settings.basic_settings.OUTPUT_PATH)
        }
    }


def setup_openai(config):
    """Setup OpenAI configuration"""
    pass


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


def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pickle(obj, path):
    """Save object to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_json(path):
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)