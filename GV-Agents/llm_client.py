import os
from dotenv import load_dotenv
from data_structures import ClientConfig
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class LLMClient:
    """Unified client for OpenRouter and HuggingFace Transformers for chatting with LLMs."""
    def __init__(self, client_config: ClientConfig):
        """Initializes the client"""
        self.backend = client_config.backend
        self.model = client_config.model

        if self.backend == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.pipelines import pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model = AutoModelForCausalLM.from_pretrained(self.model)
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        elif self.backend == "sync_openrouter":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        elif self.backend == "async_openrouter":
            from openai import AsyncOpenAI

            print(os.getenv("OPENROUTER_API_KEY"))
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        logger.info(f"Initialized LLMClient with backend {self.backend} and model {self.model}")

    def chat(self, messages, **kwargs):
        """Chat completion given message template"""
        if self.backend == "hf":
            output = self.generator(messages, **kwargs)[0]["generated_text"]
            return output
        elif self.backend == "sync_openrouter":
            temperature = kwargs.pop("temperature", 0.0)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        else:
            raise NotImplementedError(f"Backend {self.backend} does not support synchronous chat")
    
    async def async_chat(self, messages, **kwargs):
        """Async chat completion given message template"""
        if self.backend == "async_openrouter":
            temperature = kwargs.pop("temperature", 0.0)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                **kwargs
            )
            return response.choices[0].message.content
        else:
            raise NotImplementedError(f"Backend {self.backend} does not support asynchronous chat")

if __name__ == "__main__":
    from huggingface_hub import login
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    messages = [{"role": "user", "content": "Hello, World!"}]
    
    client = LLMClient(ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"))
    print(client.chat(messages, temperature=0.0))
    
    client = LLMClient(ClientConfig("hf", "google/gemma-3-1b-it"))
    print(client.chat(messages, temperature=0.0))