# app.py
from flask import Flask, request, jsonify
from slack_sdk import WebClient
import os

app = Flask(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")  # set this in terminal or .env
client = WebClient(token=SLACK_BOT_TOKEN)

@app.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.json

    # Step 1: Handle Slack URL verification
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    # Step 2: Handle message events
    if data.get("event", {}).get("type") == "message" and "bot_id" not in data["event"]:
        channel = data["event"]["channel"]
        text = data["event"]["text"]

        # Respond back
        client.chat_postMessage(channel=channel, text=f"You said: {text}")

    return '', 200

@app.route('/')
def home():
    return "Slack bot is running!"

if __name__ == '__main__':
    app.run(port=3000, debug=True)
