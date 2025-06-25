import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Unified client for OpenRouter and HuggingFace Transformers for chatting with LLMs."""
    def __init__(self, backend: str, model: str):
        """Initializes the client"""
        self.backend = backend
        self.model = model

        if backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model)
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        elif backend == "openrouter":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def chat(self, messages, **kwargs):
        """Chat completion giving message template"""
        if self.backend == "transformers":
            output = self.generator(messages, **kwargs)[0]["generated_text"]
            return output
        elif self.backend == "openrouter":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 1.0)
            )
            return response.choices[0].message.content

if __name__ == "__main__":
    from huggingface_hub import login
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    messages = [{"role": "user", "content": "Hello, World!"}]
    
    client = LLMClient("openrouter", "google/gemma-3-27b-it")
    print(client.chat(messages, temperature=0.0))
    
    client = LLMClient("transformers", "google/gemma-3-27b-it")
    print(client.chat(messages, temperature=0.0))