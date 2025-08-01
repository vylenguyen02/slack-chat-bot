import os
import asyncio
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from service.read_message import read_message
from service.ai_processing import FileHandlingService
from service.audio_processing import audio_processing
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

load_dotenv()
# FIXED PARAM DECLATION
# Mongo + Langchain setup
client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
db = os.environ["DB_NAME"]
collection_name = os.environ["COLLECTION_NAME"]
search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]
collection = client[db][collection_name]

# Set up embedder to embed documents
embedder = OpenAIEmbeddings(
    model="azure-text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# Set up GenAI 
llm = ChatOpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"]
)

self_service = FileHandlingService(llm, "docs/", embedder, collection, search_index)

    # Optional: Load and embed new PDFs
async def main():
    if self_service.folder_has_any_file():
        for file in os.listdir(self_service.pdf_path):
            # Join the file name and folder to parse.
            new_file_path = os.path.join(self_service.pdf_path, file)
            print(new_file_path)
            if os.path.isfile(new_file_path):
                if file[-3:] == "mp3":
                    msg = stt #audio_processing(new_file_path)
                # Load file
                else:
                    docs = await self_service.loading(new_file_path)
                    # Split file
                    all_splits = self_service.splitting(docs)
                    # Embed file
                    self_service.embed_documents(all_splits, embedder, collection, search_index)
                    # Remove file so that the program does not have to load it everytime main runs.
                    # Also, remove file processsing procedure and avoid repitition.
                os.remove(new_file_path)
    else:
        return 0
if __name__ == "__main__":
    asyncio.run(main())