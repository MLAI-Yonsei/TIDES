# TIDES/src/utils/model_utils.py
import os
import time
import re
from together import Together

class ModelManager:
    def __init__(self, api_key):
        os.environ['TOGETHER_API_KEY'] = api_key
        self.client = Together()
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    def call_with_retry(self, message, max_tokens=1000, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": message}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                self._handle_api_error(e, attempt, max_retries)

    def _handle_api_error(self, error, attempt, max_retries):
        error_message = str(error)
        wait_time = self._get_wait_time(error_message)
        
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1}/{max_retries} failed. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            raise error

    def _get_wait_time(self, error_message):
        if "rate_limit_exceeded" in error_message.lower():
            wait_time_match = re.search(r"Please try again in (\d+\.?\d*)s", error_message)
            return float(wait_time_match.group(1)) if wait_time_match else 60
        return 60