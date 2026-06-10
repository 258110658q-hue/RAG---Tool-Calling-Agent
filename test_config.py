"""
配置模块验证脚本
==================
运行方式：在项目根目录下执行
    python test_config.py

如果一切正常，你会看到各项配置的读取结果。
如果 API Key 显示为空，说明 .env 文件没有正确配置。
"""

# 把 src 目录加入 Python 的搜索路径
# 因为 test_config.py 在根目录，而 config 模块在 src/ 子目录里
import sys
sys.path.insert(0, ".")

from src.config import config

print("=" * 50)
print("配置模块验证")
print("=" * 50)

# 读取 LLM 相关配置
print(f"API Key      : {config.llm_api_key[:20]}...（已隐藏后续字符）"
      if len(config.llm_api_key) > 20
      else f"API Key      : [未设置或长度不足]")

print(f"Base URL     : {config.llm_base_url}")
print(f"默认模型      : {config.llm_model}")

# 读取应用配置
print(f"最大重试次数   : {config.max_retries}")
print(f"请求超时(秒)   : {config.request_timeout}")

print("=" * 50)

# 核心验证：API Key 不能为空
if len(config.llm_api_key) < 10:
    print("⚠️  警告：API Key 似乎没有正确设置！")
    print("   请检查 .env 文件中的 LLM_API_KEY 值。")
else:
    print("✅ 配置模块工作正常，准备进入下一步。")
