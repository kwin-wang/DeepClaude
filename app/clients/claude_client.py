"""Claude API 客户端"""
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient


class ClaudeClient(BaseClient):
    # 定义 OpenAI API 兼容的 provider 列表
    OPENAI_COMPATIBLE_PROVIDERS = ["openrouter", "oneapi"]
    
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages", provider: str = "anthropic", is_openai_compatible: bool = False):
        """初始化 Claude 客户端
        
        Args:
            api_key: Claude API密钥
            api_url: Claude API地址
            is_openrouter: 是否使用 OpenRouter API
        """
        super().__init__(api_key, api_url)
        self.provider = provider
        self.is_openai_compatible = is_openai_compatible or provider in self.OPENAI_COMPATIBLE_PROVIDERS

    async def stream_chat(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        model: str,
        stream: bool = True
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式或非流式对话
        
        Args:
            messages: 消息列表
            model_arg: 模型参数元组[temperature, top_p, presence_penalty, frequency_penalty]
            model: 模型名称。如果是 OpenRouter, 会自动转换为 'anthropic/claude-3.5-sonnet' 格式
            stream: 是否使用流式输出，默认为 True
            
        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "answer"
                内容: 实际的文本内容
        """
        # OpenRouter 特殊处理
        if self.provider == "openrouter":
            model = "anthropic/claude-3.5-sonnet"
        
        # 根据 provider 设置请求头和数据
        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream" if stream else "application/json",
            }

            data = {
                "model": model,
                "messages": messages,
                "max_tokens": 8192,
                "stream": stream,
                "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
                "top_p": model_arg[1]
            }
        elif self.is_openai_compatible:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.provider == "openrouter":
                headers.update({
                    "HTTP-Referer": "https://github.com/ErlichLiu/DeepClaude",
                    "X-Title": "DeepClaude"
                })

            data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
                "top_p": model_arg[1],
                "presence_penalty": model_arg[2],
                "frequency_penalty": model_arg[3]
            }
        else:
            raise ValueError(f"不支持的Claude Provider: {self.provider}")

        logger.debug(f"开始对话：{data}")

        if stream:
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                for line in chunk_str.split('\n'):
                    if line.startswith('data: '):
                        json_str = line[6:]
                        if json_str.strip() == '[DONE]':
                            return
                            
                        try:
                            data = json.loads(json_str)
                            logger.debug(f"收到的数据: {data}")
                            
                            if self.is_openai_compatible:
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield "answer", content
                                else:
                                    logger.warning(f"收到的数据中没有choices: {data}")
                            elif self.provider == "anthropic":
                                if data.get('type') == 'content_block_delta':
                                    content = data.get('delta', {}).get('text', '')
                                    if content:
                                        yield "answer", content
                            else:
                                raise ValueError(f"不支持的Claude Provider: {self.provider}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {e}, 原始数据: {json_str}")
                        except Exception as e:
                            logger.error(f"处理数据时发生错误: {e}, 数据: {data}")
        else:
            async for chunk in self._make_request(headers, data):
                try:
                    response = json.loads(chunk.decode('utf-8'))
                    logger.debug(f"收到的非流式数据: {response}")
                    
                    if self.provider in self.OPENAI_COMPATIBLE_PROVIDERS:
                        choices = response.get('choices', [])
                        if choices:
                            message = choices[0].get('message', {})
                            content = message.get('content', '')
                            if content:
                                yield "answer", content
                        else:
                            logger.warning(f"收到的数据中没有choices: {response}")
                    elif self.provider == "anthropic":
                        content_list = response.get('content', [])
                        if content_list:
                            content = content_list[0].get('text', '')
                            if content:
                                yield "answer", content
                        else:
                            logger.warning(f"收到的数据中没有content: {response}")
                    else:
                        raise ValueError(f"不支持的Claude Provider: {self.provider}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}, 原始数据: {chunk.decode('utf-8')}")
                except Exception as e:
                    logger.error(f"处理数据时发生错误: {e}, 数据: {response}")
