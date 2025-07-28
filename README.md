# Slack Chatbot with LangChain, MongoDB Atlas, and OpenAI
This project implements a Slack chatbot that uses semantic search and generative AI to retrieve relevant documents and provide intelligent answers. It leverages the power of LangChain, MongoDB Atlas Vector Search, and OpenAI's ChatGPT to understand context, search for similar documents, and respond concisely.

## 🔧 Features
💬 Reads messages from Slack in real-time

🧠 Understands natural language queries

🔍 Retrieves top-matching documents using vector similarity

✨ Generates concise responses based on retrieved context

⚙️ Configurable pipeline with LangChain and LangGraph

## 🧱 Project Structure
```bash
slack-chat-bot/
├── main.py                      # Entry point for the chatbot
├── service/
│   ├── read_message.py          # Slack event handling and messaging
│   ├── ai_processing.py         # Retrieval and generation logic
│   └── text_processing.py       # Helper utilities (optional)
├── .env                         # Environment variables (API keys, Slack tokens)
├── requirements.txt             # Python dependencies
└── README.md                    # You're here!
```

## 🚀 How It Works
### 1. Slack Message Triggered
The bot listens for new messages in a configured Slack channel using Slack Events API.

### 2. Routing to AI
Each message is passed through a decision node to determine whether retrieval is needed.

### 3. Semantic Search
If needed, it performs vector similarity search using MongoDB Atlas to find relevant documents.

### 4. Answer Generation
Using a retrieval-augmented generation approach (RAG), the LLM (OpenAI) crafts a final response based on the retrieved context.

## 🛠️ Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/slack-chat-bot.git
cd slack-chat-bot
```

### 2. Create env file
``` bash 
env file
Copy code
OPENAI_API_KEY=your_openai_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_APP_TOKEN=your_slack_app_token
CONVERSATION_ID=your_channel_or_thread_id
MONGODB_URI=your_mongo_connection_string
```

### 3. Install Dependencies
``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Bot
``` bash
python main.py
```

## 🧠 Core Functions
### make_retrieve()
Creates a tool for semantic document retrieval using MongoDB Atlas Vector Search.

### make_query_or_respond()
Decides whether to retrieve documents or let the model respond directly.

### generate()
Uses the retrieved context and LangChain LLM to craft a final answer.

### graph_building()
Creates a LangGraph state machine to handle message flow and transitions.

📦 Dependencies
``` bash langchain```
``` bash openai```
``` bash slack_sdk```
``` bash pymongo```
``` bash python-dotenv```
``` bash langchain_mongodb```
```bash langgraph```


