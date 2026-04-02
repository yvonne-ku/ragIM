import streamlit as st
import requests
import time

# 设置页面标题和图标
st.set_page_config(
    page_title="RagIM - 智能对话系统",
    page_icon="💬",
    layout="wide"
)

# API 端点
API_URL = 'http://localhost:8000/chat'

# 页面标题
st.title("RagIM - IM 对话智能辅助系统")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 RagIM 智能助手，有什么可以帮助你的吗？"}
    ]

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("输入你的问题..."):
    # 添加用户消息到会话状态
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手消息（加载状态）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("正在思考...")

        try:
            # 调用 API
            response = requests.post(
                API_URL,
                json={"message": prompt},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                bot_response = data.get("response", data.get("message", "抱歉，我无法理解你的问题。"))
            else:
                bot_response = f"API 请求失败，状态码：{response.status_code}"

        except Exception as e:
            bot_response = f"请求失败：{str(e)}"

        # 更新助手消息
        message_placeholder.markdown(bot_response)

    # 添加助手回复到会话状态
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# 侧边栏
with st.sidebar:
    st.header("关于 RagIM")
    st.markdown("RagIM 是一个基于知识检索的智能对话系统，针对企业 IM 多模态、碎片化、实时性的对话数据构建，提供对话信息的智能问答服务，推动企业内部的主题知识发现。")
    st.markdown("---")
    st.header("使用说明")
    st.markdown("1.在输入框中输入你的问题<br>2.等待系统生成回答<br>3.继续提问进行多轮对话", unsafe_allow_html=True)
    st.markdown("---")
    st.header("API 配置")
    api_url = st.text_input("API 端点", value=API_URL)
    if api_url != API_URL:
        API_URL = api_url
    st.info("API 端点已更新")