from service.read_message import read_message
import asyncio
from service.ai_processing import FileHandlingService

from langchain_openai import OpenAIEmbeddings  
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import uuid
from langchain_core.messages import SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory
from service.ai_processing import FileHandlingService
from service.element import Element
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from slack_sdk import WebClient

load_dotenv()



# TOOLS DECLARATION
client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
db = os.environ["DB_NAME"]
collection_name = os.environ["COLLECTION_NAME"]
search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]
collection = client[db][collection_name]
conversation_id = os.environ["CONVERSATION_ID"]
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

vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,  # same embedding model used when saving
            index_name=search_index,
            relevance_score_fn="cosine"
        )

pdf_path = "docs/"

async def main():
    # PART 1: DATA PROCESSING
    self_service = FileHandlingService(llm, "docs/", embedder, collection, search_index)
    if self_service.folder_has_any_file():
            for file in os.listdir(self_service.pdf_path):
                new_file_path = os.path.join(self_service.pdf_path, file)
                if os.path.isfile(new_file_path):
                    docs = await self_service.loading(new_file_path)
                    all_splits = self_service.splitting(docs)
                    vectorstore = self_service.embed_documents(all_splits, embedder, collection, search_index)
                    os.remove(new_file_path)
    else:
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,  # same embedding model used when saving
            index_name=search_index,
            relevance_score_fn="cosine"
        )
    # PART 2: TAKE USER INPUT
    input_message = read_message()

    # PART 3: SAVE USER INPUT ON THE CLOUD
    import asyncio


async def main():
    self_service = FileHandlingService(llm, "docs/", embedder, collection, search_index)
    if self_service.folder_has_any_file():
            for file in os.listdir(self_service.pdf_path):
                new_file_path = os.path.join(self_service.pdf_path, file)
                if os.path.isfile(new_file_path):
                    docs = await self_service.loading(new_file_path)
                    all_splits = self_service.splitting(docs)
                    vectorstore_db = self_service.embed_documents(all_splits, embedder, collection, search_index)
                    os.remove(new_file_path)
    else:
        vectorstore_db = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,  # same embedding model used when saving
            index_name=search_index,
            relevance_score_fn="cosine"
        )
    
    message = read_message()
    if message == 0:
        return 0
    tools = ToolNode([self_service.make_retrieve(vectorstore_db)])
    

    graph = self_service.graph_building(tools, vectorstore_db)
    for step in graph.stream(
        { "messages": [{"role": "user", "content": message}]},
        stream_mode="values",
    ):
        for msg in step["messages"]:
            if isinstance(msg, AIMessage):
                final_ai_message = msg.content

    slack_client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])

    response = slack_client.chat_postMessage(
        channel=conversation_id,
        text=final_ai_message
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())