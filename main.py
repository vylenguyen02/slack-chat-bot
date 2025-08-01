import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from slack_sdk import WebClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.messages import AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient
from service.text_related_processing import TextProcessing, ConversationGraphBuilder, DocProcessing
from service.audio_processing import AudioProcessing
from service.picture_processing import PictureProcessing
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt import App


import asyncio
load_dotenv()
app = App(token=os.environ["SLACK_APP_TOKEN"])

def folder_has_any_file(pdf_path):
        for f in os.listdir(pdf_path):
            if os.path.isfile(os.path.join(pdf_path, f)) and not f.startswith("."):
                return True
        return False

async def process_and_respond(user_message):
    # INSTANCE DECLARATION
    client_mongodb = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
    db = os.environ["DB_NAME"]
    collection_name = os.environ["COLLECTION_NAME"]
    search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]
    collection = client_mongodb[db][collection_name]

    llm = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"]
    )

    embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
    )

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedder,
        index_name=search_index,
        relevance_score_fn="cosine"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    pdf_path = "docs/"
    # LOAD FILES
    if folder_has_any_file(pdf_path):
        for file in os.listdir(pdf_path):
            new_file_path = os.path.join(pdf_path, file)
            if os.path.isfile(new_file_path):
                if "mp3" in new_file_path:
                    audio_processing = AudioProcessing(new_file_path)
                    transcription = audio_processing.audio_processing(file)
                    text_loading = TextProcessing(text_splitter, embedder, collection, search_index)
                    vectorstore = text_loading.splitting(transcription)
                elif "pdf" in new_file_path:
                    pdf_loading = DocProcessing(new_file_path, text_splitter, embedder, collection, search_index)
                    vectorstore = await pdf_loading.loading(file)
                else:
                    print("Invalid type of file.")
                    # Remove file so that the program does not have to load it everytime main runs.
                    # Also, remove file processsing procedure and avoid repitition.
            os.remove(new_file_path)
    conversation_id = os.environ["CONVERSATION_ID"]
    client_slack = WebClient(token=os.environ["BOT_USER_OAUTH_TOKEN"])


    result = client_slack.conversations_history(
            channel= conversation_id,
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

    # check for input type
    if message.get('files') == None: # when input is text
        read_user_input = message.get('text') 

    else:
    # if input is file-upload-related
        if message.get('files')[0].get('mimetype')[:5] == 'audio': # check if mimetype key is audio
            audio_processing = AudioProcessing(message.get('files')[0].get('url_private_download'))
            read_user_input = audio_processing.audio_download()
            # print(transcription)
        elif message.get('files')[0].get('mimetype')[:5] == 'image':
            image_processing = PictureProcessing(client_slack, conversation_id, embedder, llm, message.get('text'), message.get('files')[0].get('url_private_download'))
            image_processing.ai_vision()
            return
        # elif "pdf" in message.get('files')[0].get('mimetype'):
        #     pdf_downloading = PDFDownloading(message.get('files')[0].get('url_private_download'))
        #     doc_processing = DocProcessing("files/", text_splitter, embedder, collection, search_index)
        #     vectorstore = await doc_processing.loading("pdf_file.pdf")

        else:
            print("File not supported.")
            return 
    print("a")
    print(read_user_input)
    print("b")
    vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,
            index_name=search_index,
            relevance_score_fn="cosine"
        )
    self_service = ConversationGraphBuilder(llm)

    tools = ToolNode([self_service.make_retrieve(vectorstore)])
    graph = self_service.graph_building(vectorstore, tools)
        # Start streaming the processing of a user message through the graph.
        # It takes a dictionary with a list of messages, starting with the user message.
    for step in graph.stream(
    {"messages": [{"role": "user", "content": read_user_input}]},
    stream_mode="values",   # This mode yields only the values (not full metadata).
    ):
        # For each step output, look for messages returned by tools or the AI.
        for msg in step.get("messages", []):
            if isinstance(msg, AIMessage):
                final_answer = msg.content
                print(final_answer)
    
    # Get final answer from AI Message
    hello = final_answer

    # Post message in slack
    response = client_slack.chat_postMessage(
        channel="#general",
        text=hello
    )
    print(response)
    # client_mongodb.drop_database(db)
    return 0

@app.event("message")

# handle_message_events
# This function is automatically triggered whenever a message event is received in Slack.
# It filters out messages from bots (including itself), logs the message, and calls the main processing function.
# Params:
# - body: the full event payload from Slack (contains event metadata and message content)
# - logger: a logger provided by Slack Bolt to log info or errors
def handle_message_events(body, logger):
    event = body.get("event", {})
    user = event.get("user")
    bot_id = event.get("bot_id")
    text = event.get("text")

    # Prevent the bot from responding to its own messages or other bots
    if bot_id or user is None:
        return

    logger.info(f"Received message: {text}")
    asyncio.run(process_and_respond(text))

# Start bot
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
