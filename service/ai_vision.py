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

# Fetch image data
def ai_vision(pic, message):
    file_info = pic.get('files')[0]
    url = file_info.get("url_private_download")
    headers = {"Authorization": f"Bearer {os.environ['BOT_USER_OAUTH_TOKEN']}"}

    response = requests.get(url, headers=headers)
    # Convert image content to base64 directly
    picture_loading = base64.b64encode(response.content).decode("utf-8")

    if message == "":
        message = "Describe the picture"
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

    response = llm.invoke([respond])
    response = slack_client.chat_postMessage(
        channel=conversation_id,
        text=response.text()
    )
    return 0
    # return response