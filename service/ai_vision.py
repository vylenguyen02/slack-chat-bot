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

# PARAM DECLARATION
slack_client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])
conversation_id = os.environ["CONVERSATION_ID"]
client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])
embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['AZURE_OPENAI_ENDPOINT']
    )
llm = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"]
    )

# ai_vision
# This function takes user input and fetch image data
# Param: 
# pic - jpeg or png file. 
# message - string (optional), empty if no action requirement is needed
# Return: None, but it posts message to channel.
def ai_vision(pic, message):

   # Get metadata for the first uploaded file in the Slack message
    file_info = pic.get('files')[0]

    # Get the private download URL for the file (only accessible with auth)
    url = file_info.get("url_private_download")

    # Prepare the authorization header using the Slack bot token
    headers = {"Authorization": f"Bearer {os.environ['BOT_USER_OAUTH_TOKEN']}"}

    # Get information from the website
    response = requests.get(url, headers=headers)
    # Convert image content to base64 directly
    picture_loading = base64.b64encode(response.content).decode("utf-8")

    # check if the requirement is included in the picture.
    if message == "":

        # Default action requirement.
        message = "Describe the picture. Only describe the contents of the image. Do not greet or ask questions. Do not say 'Hello. How can I help you today?'."
    
    # Generate answer based on picture and requirement.
    respond = {
        "role": "user",
        "content": [
            {"type": "text", "text": message},
            {
                "type": "image",
                "source_type": "base64",
                "data": picture_loading,
                "mime_type": "image/jpeg",  # or image/png depending on file type
            },
        ],
    }

    # asking GenAI to generate response
    response = llm.invoke([respond])

    # post message in channel directly 
    response = slack_client.chat_postMessage(
        channel=conversation_id,
        text=response.text()
    )

    # returning 0, because response is not returnable.
    return 0
