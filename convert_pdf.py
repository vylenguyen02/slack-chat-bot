from slack_sdk import WebClient
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def read():
    conversation_id = os.environ["CONVERSATION_ID"]
    client_slack = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])
    result = client_slack.conversations_history(
                channel= conversation_id,
                inclusive=True,
                limit=12
            )
    comment = result["messages"][0]
    # print(response)
    url = comment.get("files")[0].get("url_private_download")
    print(url)

    response = requests.get(url)
    if response.status_code == 200:
        with open("try.pdf", 'wb') as f:
            f.write(response.content)
        print(f"[✔] File downloaded to file/")
    else:
        raise Exception("Failed to download file")

read()