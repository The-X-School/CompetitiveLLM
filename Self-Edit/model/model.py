from openai import OpenAI

OPENROUTER_API_KEY = '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'
model = "deepseek/deepseek-r1:free"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)