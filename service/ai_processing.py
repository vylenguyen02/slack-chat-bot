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

class FileHandlingService:
    def __init__(self, llm, pdf_path, embedder, collection, search_index):
        self.llm = llm
        self.pdf_path = os.path.abspath(pdf_path)
        self.embedder = embedder
        self.collecton = collection
        self.search_index = search_index

    def folder_has_any_file(self):
        for f in os.listdir(self.pdf_path):
            if os.path.isfile(os.path.join(self.pdf_path, f)) and not f.startswith("."):
                return True
        return False

    async def loading(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs
    
    def splitting(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
        all_splits = text_splitter.split_documents(docs)
        return all_splits

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

    def make_retrieve(self, vectorstore):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vectorstore.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        return retrieve 
    
    def make_query_or_respond(self, vectorstore):   
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([self.make_retrieve(vectorstore)])
            response = llm_with_tools.invoke(state["messages"])
            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}
        return query_or_respond

    def generate(self, state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def graph_building(self, tools, vectorstore_db):
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(self.make_query_or_respond(vectorstore_db))
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile()
        return graph
# Define state for application