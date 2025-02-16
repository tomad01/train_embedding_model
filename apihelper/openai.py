import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)



class Gpt4oWrapper:
    def __init__(self,mini=False):
        if mini:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_GPT4oMINI_KEY"),
                api_version="2024-08-01-preview",
                azure_endpoint=os.getenv("AZURE_GPT4oMINI_ENDPOINT"),
            )
            self.model_name = "gpt-4o-mini"
        else:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_GPT4o_KEY"),
                api_version="2024-08-01-preview",
                azure_endpoint=os.getenv("AZURE_GPT4o_ENDPOINT"),
            )
            self.model_name = "gpt-4o"


    
    def generate(self,prompt,text):
        messages = [
            {"role": "system","content": prompt},
            {"role": "user","content": text}
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,messages=messages,
            max_tokens=512,
        )

        return response.choices[0].message.content