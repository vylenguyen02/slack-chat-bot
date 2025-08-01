from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
import json

import requests

load_dotenv()

# Audio Processing class
# This class processes audio files by 
class AudioProcessing:
    def __init__(self, url_priv_download):
        self.url_priv_download = url_priv_download

    def audio_download(self):
        headers = {
        "Authorization": f"Bearer {os.environ["BOT_USER_OAUTH_TOKEN"]}"
    }
        response = requests.get(self.url_priv_download, headers=headers)
        with open("audio.mp3", "wb") as f:
            f.write(response.content)
        return self.audio_processing("audio.mp3")
    
    def audio_processing(self, audio):
        client = OpenAI(
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
        with open("audio.mp3", "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
            model="azure-whisper", 
            file=audio_file,
            response_format="json"
        )
        print(transcription.text)
        return transcription.text
