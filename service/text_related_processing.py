from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END

class DocProcessing:
    def __init__(self, pdf_path, text_splitter, embedder, collection, search_index):
        self.pdf_path = pdf_path
        self.text_splitter = text_splitter
        self.embedder = embedder
        self.collection = collection
        self.search_index = search_index
    
    async def loading(self, filename):
            loader = PyPDFLoader(self.pdf_path)
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)
            return self.splitting(docs)
    
    def splitting(self, docs):
        all_splits = self.text_splitter.split_documents(docs)
        return self.embedding(all_splits)
            
    def embedding(self, all_splits):
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
                documents=all_splits,
                embedding=self.embedder,
                collection=self.collection,
                index_name=self.search_index
            )
        ids=[str(val) for val in range(len(all_splits))],
        vectorstore.create_vector_search_index(dimensions=3072)
        vectorstore.add_documents(all_splits,ids=ids[0])
        return vectorstore

class TextProcessing:
    def __init__(self, text_splitter, embedder, collection, search_index):
        self.text_splitter = text_splitter
        self.embedder = embedder
        self.collection = collection
        self.search_index = search_index

    def splitting(self, text):
        all_splits = self.text_splitter.split_text(text)
        return self.embedding(all_splits)

    def embedding(self, all_splits):
        vectorstore = MongoDBAtlasVectorSearch(
            embedding=self.embedder,
            collection=self.collection,
            index_name=self.search_index
        )
        vectorstore.create_vector_search_index(dimensions=3072)
        vectorstore.add_texts(all_splits)
        return vectorstore

    
class ConversationGraphBuilder:
    def __init__(self, llm):
        self.llm = llm
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
    def graph_building(self, vectorstore, tools):
        # Initialize state graph
        graph_builder = StateGraph(MessagesState)

        # Add nodes for decision, tool use, and final generation
        graph_builder.add_node(self.make_query_or_respond(vectorstore))
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