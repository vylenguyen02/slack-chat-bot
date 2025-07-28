from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from typing import TypedDict, List
from langchain.schema import Document
from langchain import hub
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# FileHandlingService
# The class processes files and generate answers for user's input
class FileHandlingService:
    # The class handles file by loading, splitting, embedding, and retrieving.
    # The class then generate tool call to generate questions with retrieval or respond 
    # Lastly, the class uses graph to 

    # Attributes:
        # llm: A boolean indicating if we like SPAM or not.
        # pdf_path: A string pointing to the folder storing files
        # embedder: An object that declares which AI tool will be used in this program
        # collection: A string containing the name of collection storing our embedded files
        # search_index: A string containing the name of our search index on MongoDBAtlas

    def __init__(self, llm, pdf_path, embedder, collection, search_index):
        self.llm = llm
        self.pdf_path = os.path.abspath(pdf_path)
        self.embedder = embedder
        self.collecton = collection
        self.search_index = search_index

    # folder_has_any_file
    # The function checks if there is any file in the given folder path
    # Param: None
    # Return: A boolean 
    def folder_has_any_file(self):
        for f in os.listdir(self.pdf_path):
            if os.path.isfile(os.path.join(self.pdf_path, f)) and not f.startswith("."):
                return True
        return False

    # loading
    # The function load the information in the file into chunks in a list
    # Param: file_path - The string stating the location of the folder containing the file
    # Return: A list containing values from the file
    async def loading(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs
    
    # splitting
    # The function splits the document into smaller chunks following by the "langchain.text_splitter" library 
    # Param: docs - a list with values from the document
    # Return: all_splits - a list of documents
    def splitting(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
        all_splits = text_splitter.split_documents(docs)
        return all_splits

    # embed_documents
    # The function stores the embeddings into a MongoDB collection.
    # Param: 
    # - all_splits: a List of Documents
    # - embedder: an object that declares which AI tool will be used in this program
    # - collection: a string containing the name of collection on MongoDB cloud.
    # - search_index: a string storing the name of search_index on the cloud.
    # Return: vectorstore -  MongoDBAtlasVectorSearch vectorstore from the given document chunks.
    def embed_documents(self, all_splits, embedder, collection, search_index):
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
                    documents=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
        ids=[str(val) for val in range(len(all_splits))],
        vectorstore.create_vector_search_index(dimensions=3072)
        vectorstore.add_documents(all_splits,ids=ids[0])
        return vectorstore

    # make_retrieve

    # make_retrieve
    # This function defines a tool for document retrieval using semantic similarity.
    # Param:
    # vectorstore - a vector search database (MongoDBAtlasVectorSearch)
    # Return: a callable tool function that retrieves top 2 most similar documents.
    def make_retrieve(self, vectorstore):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            # Perform similarity search on the vectorstore
            retrieved_docs = vectorstore.similarity_search(query, k=2)

            # Format retrieved documents into a string
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return retrieve


    # make_query_or_respond
    # This function decides whether to perform retrieval or respond directly.
    # Param:
    # vectorstore - a vector search database (MongoDBAtlasVectorSearch)
    # Return: a callable function that invokes the LLM with tools
    def make_query_or_respond(self, vectorstore):   
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            # Bind LLM with the retrieval tool
            llm_with_tools = self.llm.bind_tools([self.make_retrieve(vectorstore)])
            
            # Run inference on message state
            response = llm_with_tools.invoke(state["messages"])

            # Wrap response in expected format
            return {"messages": [response]}

        return query_or_respond


    # generate
    # This function generates a final answer using context from retrieved documents.
    # Param:
    # state - MessagesState containing the conversation and tool outputs
    # Return: dict containing one AI-generated message
    def generate(self, state: MessagesState):
        """Generate answer."""
        # Collect recent tool messages (retrieved docs)
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format retrieved content into a prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. Do not use any information beside the docs given."
            "\n\n"
            f"{docs_content}"
        )

        # Filter the conversation to keep human/system/AI messages only
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Generate response
        response = self.llm.invoke(prompt)
        return {"messages": [response]}


    # graph_building
    # This function builds a LangGraph that defines how the chatbot processes messages.
    # Param:
    # tools - ToolNode that performs retrieval
    # vectorstore_db - the vector database used for searching documents
    # Return: compiled graph object
    def graph_building(self, tools, vectorstore_db):
        # Initialize state graph
        graph_builder = StateGraph(MessagesState)

        # Add nodes for decision, tool use, and final generation
        graph_builder.add_node(self.make_query_or_respond(vectorstore_db))
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)

        # Set entry point and transitions
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        # Compile and return the graph
        graph = graph_builder.compile()
        return graph

