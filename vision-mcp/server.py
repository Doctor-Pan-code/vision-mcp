"""
Vision MCP Server - 视觉能力代理服务器
========================================
功能：让不具备视觉能力的 LLM 通过此 MCP 调用视觉模型分析图片。
后端：支持任何 OpenAI 兼容的视觉模型 API（硅基流动 / OpenAI / DashScope / Ollama 等）。

每个部署者自行选择视觉模型提供商，通过环境变量配置自己的 API Key。

API Key 配置方式（按优先级）：
  1. 环境变量 VISION_API_KEY（魔搭部署时通过平台配置，最安全）
  2. 兼容旧版环境变量 SILICONFLOW_API_KEY
  3. .env 文件（本地开发用，不会提交到代码仓库）

使用方式：
  set VISION_API_KEY=sk-xxx && set VISION_PROVIDER=siliconflow
  python server.py                    # stdio 模式（本地 MCP 客户端）
  set VISION_MCP_TRANSPORT=sse
  python server.py                    # SSE 模式（魔搭远程部署）
"""

import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# ============================================================
# 1. 加载环境配置
# ============================================================

load_dotenv()

# ============================================================
# 2. 提供商预设
# ============================================================
# 每个人部署时可自由选择视觉模型提供商：
#   - siliconflow : 硅基流动（国内，免费额度，推荐中文用户）
#   - openai      : OpenAI GPT-4o（海外，需国际支付）
#   - dashscope   : 阿里云通义千问 VL（国内，需阿里云 AK）
#   - custom      : 自定义 OpenAI 兼容接口（Ollama / vLLM / 任意中转）
#
# 只需设置 VISION_PROVIDER 即可自动填入 BASE_URL，
# 每个用户仅需提供自己的 VISION_API_KEY。

PROVIDER_PRESETS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "description": "硅基流动（推荐，国内直接访问，免费额度）",
        "default_model": "Qwen/Qwen2-VL-72B-Instruct",
        "example_models": [
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen3-VL-72B-Instruct",
            "deepseek-ai/deepseek-vl2",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI（海外，需国际网络+信用卡）",
        "default_model": "gpt-4o",
        "example_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
        ],
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问（国内，需阿里云 AK）",
        "default_model": "qwen-vl-plus",
        "example_models": [
            "qwen-vl-plus",
            "qwen-vl-max",
            "qwen2.5-vl-72b-instruct",
        ],
    },
    "custom": {
        "base_url": None,  # 由 VISION_BASE_URL 指定
        "description": "自定义 OpenAI 兼容接口（任意 API）",
        "default_model": "gpt-4o",
        "example_models": [],
    },
}

# ============================================================
# 3. 读取环境变量（支持新旧变量名）
# ============================================================

# 提供商：siliconflow / openai / dashscope / custom
PROVIDER = os.getenv("VISION_PROVIDER", "siliconflow").lower()

# API Key：主变量 VISION_API_KEY，兼容旧版 SILICONFLOW_API_KEY
API_KEY = os.getenv("VISION_API_KEY", "") or os.getenv("SILICONFLOW_API_KEY", "")

# BASE_URL：先看 VISION_BASE_URL，再看提供商预设，最后看旧版 SILICONFLOW_BASE_URL
if PROVIDER in PROVIDER_PRESETS and PROVIDER_PRESETS[PROVIDER]["base_url"]:
    DEFAULT_BASE_URL = PROVIDER_PRESETS[PROVIDER]["base_url"]
else:
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
BASE_URL = os.getenv("VISION_BASE_URL", "") or os.getenv("SILICONFLOW_BASE_URL", "") or DEFAULT_BASE_URL

# 默认模型：按提供商预设，可被环境变量覆盖
preset = PROVIDER_PRESETS.get(PROVIDER, PROVIDER_PRESETS["siliconflow"])
DEFAULT_MODEL = os.getenv("DEFAULT_VISION_MODEL", "") or preset["default_model"]

# ============================================================
# 4. 创建 MCP 服务器
# ============================================================

mcp = FastMCP(
    "vision-mcp",
    instructions="视觉能力代理 - 让 LLM 能够分析图片内容。支持多个视觉模型提供商。",
)

# 当前正在使用的模型
_current_model = DEFAULT_MODEL


# ============================================================
# 5. 辅助函数
# ============================================================

def _get_api_key_error() -> str | None:
    """检查 API Key 是否已配置，未配置则返回错误信息"""
    if not API_KEY:
        return (
            "[错误] 未配置 VISION_API_KEY\n\n"
            "部署到魔搭：在魔搭 MCP 服务管理页面的「环境变量」中设置\n"
            "  VISION_API_KEY = 你的视觉模型 API Key\n"
            "  VISION_PROVIDER = siliconflow（或 openai / dashscope / custom）\n\n"
            "本地开发：在 vision-mcp 目录下创建 .env 文件：\n"
            "  VISION_API_KEY=你的Key\n"
            "  VISION_PROVIDER=siliconflow\n\n"
            "注册地址：\n"
            "  硅基流动: https://siliconflow.cn\n"
            "  OpenAI:   https://platform.openai.com\n"
            "  阿里云:   https://dashscope.aliyun.com"
        )
    return None


def _create_client(api_key: str, base_url: str) -> OpenAI:
    """创建 OpenAI 兼容客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# 6. 定义 MCP 工具
# ============================================================


@mcp.tool(
    name="analyze_image",
    description="""分析一张图片的内容。

向视觉模型发送图片 URL 和你的问题，模型会理解图片内容并回答。
model 参数不传时使用默认模型。不同提供商默认模型不同：
- siliconflow: Qwen/Qwen2-VL-72B-Instruct
- openai: gpt-4o
- dashscope: qwen-vl-plus
- custom: 由部署者自定

你也可以传入自己熟悉的任意模型名称（只要在你的提供商平台可用）。

适用于：
- 识别图片中的物体、场景、文字
- 回答关于图片内容的问题
- 详细描述图片内容
- 从图片中提取文字信息

注意：图片 URL 必须可公开访问。""",
)
def analyze_image(
    image_url: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """
    参数说明：
        image_url: 图片的公开访问 URL（必填）
        prompt:    你对图片提出的问题或描述要求（必填）
        model:     视觉模型名称（可选）。不传则使用当前默认模型。
    """
    global _current_model

    err = _get_api_key_error()
    if err:
        return err

    model_name = model or _current_model
    if model:
        _current_model = model

    try:
        # 每次调用创建新的客户端实例，使用当前配置
        client = _create_client(API_KEY, BASE_URL)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        result = response.choices[0].message.content
        return result or "（模型未返回任何内容）"

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return (
                "[错误] API 认证失败。请检查 VISION_API_KEY 是否正确配置。\n"
                "当前提供商: %s\n"
                "当前 API 地址: %s" % (PROVIDER, BASE_URL)
            )
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return (
                "[错误] 模型 '%s' 在当前提供商 '%s' 下不可用。\n\n"
                "请检查模型名称是否正确，或更换其他模型。\n"
                "当前 API 地址: %s" % (model_name, PROVIDER, BASE_URL)
            )
        elif "rate" in error_msg.lower():
            return "[错误] API 调用过于频繁，请稍后再试。"
        else:
            return "[错误] 调用视觉模型失败：%s" % error_msg


@mcp.tool(
    name="list_vision_models",
    description="列出当前提供商推荐使用的视觉模型列表及使用说明。",
)
def list_vision_models() -> str:
    """返回当前提供商推荐的视觉模型列表"""
    provider_info = PROVIDER_PRESETS.get(PROVIDER, PROVIDER_PRESETS["siliconflow"])
    provider_name_cn = {
        "siliconflow": "硅基流动",
        "openai": "OpenAI",
        "dashscope": "阿里云通义千问",
        "custom": "自定义接口",
    }.get(PROVIDER, PROVIDER)

    result = "## 当前视觉模型提供商\n\n"
    result += "**%s** (%s)\n" % (provider_name_cn, provider_info["description"])
    result += "API 地址: `%s`\n\n" % BASE_URL

    if provider_info["example_models"]:
        result += "### 推荐模型\n\n"
        result += "| 模型名称 | 说明 |\n"
        result += "|----------|------|\n"
        result += "| **%s** | [默认] |\n" % provider_info["example_models"][0]
        for m in provider_info["example_models"][1:]:
            result += "| %s | |\n" % m
        result += "\n"

    result += "### 使用方式\n\n"
    result += "在 `analyze_image` 工具的 `model` 参数中传入模型名称即可。\n"
    result += "完整模型列表请查看你的提供商文档。\n\n"

    if PROVIDER == "siliconflow":
        result += "硅基流动模型广场: https://siliconflow.cn/models\n"
    elif PROVIDER == "openai":
        result += "OpenAI 模型列表: https://platform.openai.com/docs/models\n"
    elif PROVIDER == "dashscope":
        result += "阿里云模型文档: https://help.aliyun.com/zh/model-studio/\n"

    return result


# ============================================================
# 7. 服务器入口
# ============================================================

def main():
    """启动 MCP 服务器"""
    print("=" * 50)
    print("  Vision MCP Server - 视觉能力代理")
    print("=" * 50)
    print("  视觉提供商: %s" % PROVIDER)
    print("  API 地址:   %s" % BASE_URL)
    print("  默认模型:   %s" % DEFAULT_MODEL)
    print("  API Key:    %s" % ("[OK] 已配置" if API_KEY else "[!] 未配置"))
    print("=" * 50)

    transport = os.getenv("VISION_MCP_TRANSPORT", "stdio")

    if transport == "sse":
        print("  启动模式: SSE (HTTP) - 适用于魔搭远程部署")
        print("=" * 50)
        mcp.run(transport="sse")
    else:
        print("  启动模式: stdio - 适用于本地 MCP 客户端配置")
        print("=" * 50)
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
