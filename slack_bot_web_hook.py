import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

# token = os.getenv("OPENAI_API_KEY") 

client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])

try:
    response = client.chat_postMessage(
        channel="#general",
        text="Hello from your bot!"
    )
    print(response)
except SlackApiError as e:
    print(f"Error sending message: {e.response['error']}")