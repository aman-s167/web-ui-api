# src/llm/gemini_llm.py
from google import genai

class GeminiLLM:
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def invoke(self, messages):
        """
        A simple implementation that concatenates the contents of all messages and sends them to Gemini.
        You may need to adjust this based on how your agent expects the response.
        """
        # Assume messages is a list of objects with a .content attribute.
        prompt = "\n".join(m.content for m in messages if hasattr(m, "content"))
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        # We return an object with a 'content' attribute for compatibility.
        # In a production implementation, you would wrap the response properly.
        return type("LLMResponse", (object,), {"content": response.text})
