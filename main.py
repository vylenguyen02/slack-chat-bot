from service.read_message import read_message, respond_message
import asyncio
from service.ai_processing import FileHandlingService
from service.text_processing import TextHandlingService
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

embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['AZURE_OPENAI_ENDPOINT']
    )

model = ChatOpenAI(
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

def main():
    # PART 1: DATA PROCESSING
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    self_service = FileHandlingService(model=model, pdf_path=pdf_path, embedder=embedder, vectorstore=vectorstore)
    if self_service.folder_has_any_file():
        for file in os.listdir(self_service.pdf_path):
            if os.path.isfile(pdf_path+file):
                docs = self_service.loading(pdf_path=pdf_path, filename=file)
                table_elements, text_elements, img_summaries= self_service.splitting(docs)
                self_service.embed_documents(prompt_text, table_elements, text_elements, img_summaries, prompt)
                os.remove(pdf_path+file)
    # PART 2: TAKE USER INPUT
    message = read_message()

    # PART 3: SAVE USER INPUT ON THE CLOUD
    text_handling = TextHandlingService()
    text = text_handling.splitting(message)
    text_handling.embed_documents(text, embedder, collection, search_index)
    
    # PART 4: PROCESS USER INPUT 
    tools = ToolNode([self_service.make_retrieve(vectorstore)])

    graph = self_service.graph_building(tools, vectorstore)
    for step in graph.stream(
    {"messages": [{"role": "user", "content": message}]},
    stream_mode="values",
):
        for msg in step.get("messages", []):
            if isinstance(msg, AIMessage):
                final_answer = msg.content
                print(final_answer)
    hello = final_answer

    slack_client = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])

    response = slack_client.chat_postMessage(
        channel="#general",
        text=hello
    )
    print(response)
    return 0


if __name__ == "__main__":
    main()