import os
from openai import OpenAI

class AIWorkflow:
    def __init__(self, config: dict, api_key: str = None):
        """
        Initialize the workflow with configuration and credentials.
        """
        self.config = config
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def process(self, user_input: str) -> str:
        """
        The execution method to be called by PFE V4.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": self.config.get("system_prompt", "")},
                    {"role": "user", "content": user_input}
                ],
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 150)
            )
            
            # Extract simple text output
            return response.choices[0].message.content.strip()

        except Exception as e:
            # Return error as string to keep pipeline flowing
            return f"Error in AI Workflow: {str(e)}"