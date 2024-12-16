# TIDES/src/utils/model_manager.py
from together import Together
from openai import OpenAI

class ModelManager:
    def __init__(self, model_type, api_key, model_name=None):
        self.model_type = model_type
        if model_type == 'together':
            self.client = Together(api_key=api_key)
            self.model_name = model_name or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        elif model_type == 'openai':
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name or "gpt-3.5-turbo-0125"

    def call_with_retry(self, message, max_tokens=1000, max_retries=5):
        for attempt in range(max_retries):
            try:
                if self.model_type == 'together':
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": message}],
                        temperature=0,
                        max_tokens=max_tokens
                    )
                else:  # openai
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": message}],
                        temperature=0
                    )
                return response.choices[0].message.content
            except Exception as e:
                self._handle_error(e, attempt, max_retries)