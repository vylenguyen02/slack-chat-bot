import requests
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from service.ai_vision import ai_vision
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
        limit=12
    )
    n = 0
    # Check if the message is sent by user
    while n >= 0:
        message = result["messages"][n]
        if "user" in message:
            break         
        else: 
            n -= 1    
    message = result["messages"][n]

    # Check for user input
    if message.get('text') != "" and message.get('files') == None:
        return message.get("text")
    else:
        return ai_vision(message, message.get('text'))
