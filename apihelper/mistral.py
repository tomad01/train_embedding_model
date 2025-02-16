import os 
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv(override=True)

class MistralWrapper:
    def __init__(self):
        self.client = ChatCompletionsClient(
            endpoint=os.getenv("AZURE_MISTRAL_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_MISTRAL_KEY"))
        )
        self.model_name = "Mistral-Large-2411-fbtnd"

    def generate(self,prompt,text):
    
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": text
                },
            ],
            "max_tokens": 512,

        }
        # Send the payload to the model
        response = self.client.complete(payload)
  
        return response.choices[0].message.content