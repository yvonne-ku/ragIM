from resources.others import chat_service

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 应用
app = FastAPI(
    title="RagIM Server",
    description="RAG Instant Messenger - 基于知识检索的智能对话系统",
    version="0.3.0.dev0"
)


# 设置 CORS
from server import settings
if settings.basic_settings.OPEN_CROSS_DOMAIN:
    # 允许跨域（开发/测试环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("CORS 已启用：允许所有跨域请求")
else:
    # 不允许跨域（生产环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],  # 空列表表示不允许任何跨域
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    print("CORS 已禁用：仅允许同源请求")

# 注册路由
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RagIM Server is running",
        "version": "0.3.0.dev0",
        "endpoints": {
            "chat": "/chat",
        }
    }
@app.post("/chat")
async def chat_endpoint(request_data: dict):
    """聊天接口"""
    return await chat_service(**request_data)


# 启动服务器
def start_server(host: str = settings.basic_settings.API_SERVER["host"], port: int = settings.basic_settings.API_SERVER["port"], reload: bool = False):
    import uvicorn
    
    print(f"Starting RagIM Server on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # 开发模式启动
    start_server(host=settings.basic_settings.API_SERVER["host"], port=settings.basic_settings.API_SERVER["port"], reload=True)
