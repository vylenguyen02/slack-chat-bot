import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import base64
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings  
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class PictureProcessing:
    def __init__(self, client_slack, conversation_id, embedder, llm, message, url):
        self.client_slack = client_slack
        self.conversation_id = conversation_id 
        self.embedder = embedder
        self.llm = llm
        self.message = message
        self.url = url
# PARAM DECLARATION

    # ai_vision
    # This function takes user input and fetch image data
    # Param: 
    # pic - jpeg or png file. 
    # message - string (optional), empty if no action requirement is needed
    # Return: None, but it posts message to channel.
    def ai_vision(self):
        # Prepare the authorization header using the Slack bot token
        headers = {"Authorization": f"Bearer {os.environ['BOT_USER_OAUTH_TOKEN']}"}

        # Get information from the website
        response = requests.get(self.url, headers=headers)
        # Convert image content to base64 directly
        picture_loading = base64.b64encode(response.content).decode("utf-8")

        # check if the requirement is included in the picture.
        if self.message == "":
            # Default action requirement.
            self.message = "Describe the picture. Only describe the contents of the image. Do not greet or ask questions. Do not say 'Hello. How can I help you today?'."
        
        # Generate answer based on picture and requirement.
        respond = {
            "role": "user",
            "content": [
                {"type": "text", "text": self.message},
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": picture_loading,
                    "mime_type": "image/jpeg",  # or image/png depending on file type
                },
            ],
        }

        # asking GenAI to generate response
        response = self.llm.invoke([respond])

        # post message in channel directly 
        response = self.client_slack.chat_postMessage(
            channel= self.conversation_id,
            text=response.text()
        )
