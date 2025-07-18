import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

# token = os.getenv("OPENAI_API_KEY") 
conversation_id = "C031LPUMS4S"
client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])

def read_message():
    # Call the conversations.history method using the WebClient
    # The client passes the token you included in initialization    
    result = client.conversations_history(
        channel=conversation_id,
        inclusive=True,
        limit=1
    )

    message = result["messages"][0]
# Print message text
    return message["text"]

def respond_message(text):
    response = client.chat_postMessage(
            channel="#general",
            text=text
    )
    print(response)