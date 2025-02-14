# my_gemini_module.py

class GeminiLLM:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def __call__(self, prompt):
        # For testing, just return a dummy response.
        return f"Dummy Gemini response for prompt: {prompt}"
