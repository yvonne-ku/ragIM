from typing import Dict, Any, Union, List, Tuple
from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msg_tuple() = ("human", "你好")
    """
    
    role: str = Field(..., description="角色：user 或 assistant")
    content: str = Field(..., description="消息内容")
    
    def to_msg_tuple(self) -> Tuple[str, str]:
        """转换为消息元组 (human/ai, content)"""
        role_mapping = {
            "assistant": "ai",
            "user": "human"
        }
        return role_mapping.get(self.role, self.role), self.content
    
    def to_msg_template(self, is_raw: bool = True) -> ChatMessagePromptTemplate:
        """
        转换为消息模板
        
        Args:
            is_raw: 是否为原始内容（添加raw标记）
        """
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        
        # 当前默认历史消息都是没有input_variable的文本
        if is_raw:
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content
        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )
    
    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        """从多种数据格式创建History对象"""
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)
        else:
            raise ValueError(f"不支持的数据格式: {type(h)}")
        
        return h
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {"role": self.role, "content": self.content}