import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from server import settings
from resources.others.utils import logger

class EntityExtractor:
    def __init__(self, model_name: str = None):
        # Get model name from scenario mapping or use default
        scenario_model = settings.api_model_settings.SCENARIO_MODELS.get("extract_entity")
        self.model_name = model_name or scenario_model or settings.api_model_settings.DEFAULT_LLM_MODEL
        
        # Get model configuration
        model_config = settings.api_model_settings.MODELS.get(self.model_name)
        platform_name = model_config.platform_name if model_config else "zhipuai"
        
        # Get platform configuration
        platform_config = None
        for platform in settings.api_model_settings.MODEL_PLATFORMS:
            if platform.platform_name == platform_name:
                platform_config = platform
                break
        
        # Get temperature from model config or use default
        temperature = model_config.temperature if model_config else 0.1
        
        # Initialize LLM
        self.llm = ChatZhipuAI(
            api_key=platform_config.api_key if platform_config else settings.api_model_settings.MODEL_PLATFORMS[0].api_key,
            model=self.model_name,
            temperature=temperature,  # 低温度保证抽取稳定性
        )
        
        self.extract_prompt = ChatPromptTemplate.from_template("""
你是一个专业的实体抽取助手。请从给定的文本（JSON 格式的消息列表）中抽取实体及其关系。

抽取规则：
1. 实体：人名、地名、组织机构、技术名词、核心概念等。
2. 关系：实体之间的交互、属性、归属等。
3. 输出格式：严格的 JSON 格式，包含 "entities" 和 "relationships" 两个列表。

示例输出：
{{
  "entities": [
    {{"name": "实体1", "type": "类型", "description": "描述"}},
    {{"name": "实体2", "type": "类型", "description": "描述"}}
  ],
  "relationships": [
    {{"source": "实体1", "target": "实体2", "relation": "关系类型", "description": "关系描述"}}
  ]
}}

待处理文本：
{text}
""")

    def extract(self, text: str) -> Dict[str, Any]:
        try:
            chain = self.extract_prompt | self.llm
            response = chain.invoke({"text": text})
            # 尝试解析 JSON
            content = response.content
            # 处理可能的 markdown 标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return {"entities": [], "relationships": []}

    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            # 将 chunk 转换为字符串
            chunk_str = json.dumps(chunk, ensure_ascii=False)
            result = self.extract(chunk_str)
            
            # 为关系添加 msg_ids 溯源
            msg_ids = [msg.get("msg_id") for msg in chunk.get("messages", []) if msg.get("msg_id")]
            for rel in result.get("relationships", []):
                rel["msg_ids"] = msg_ids
                
            all_entities.extend(result.get("entities", []))
            all_relationships.extend(result.get("relationships", []))
            
        return {
            "entities": all_entities,
            "relationships": all_relationships
        }