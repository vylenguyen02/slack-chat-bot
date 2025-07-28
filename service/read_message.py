import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from service.ai_vision import ai_vision
load_dotenv()

# FIXED PARAM DECLARTION
conversation_id = os.environ["CONVERSATION_ID"]
client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])

# read_message
# This function reads user's input
# Param: None
# Return: 
# - the value of the key 'text' in the message dictionary - string value if input is text
# - if value is picture, return None but it calls to ai_vision to process the picture.

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

    # If user input is text, return string value
    if message.get('text') != "" and message.get('files') == None:
        return message.get("text")
    
    # If user input is picture with empty text value
    elif message.get('text') == "":
        return ai_vision(message, "")
    
    # If user input is picture with text value
    else:
        return ai_vision(message, message.get('text'))
