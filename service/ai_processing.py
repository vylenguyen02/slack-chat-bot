import os
import uuid
from unstructured.partition.pdf import partition_pdf
from service.element import Element
import glob
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langchain.storage import InMemoryStore
from langchain_core.tools import tool


class FileHandlingService:
    def __init__ (self, pdf_path, model, embedder, vectorstore):
        self.pdf_path = pdf_path
        self.model = model
        self.embedder = embedder
        self.vectorstore = vectorstore

    def folder_has_any_file(self):
            for f in os.listdir(self.pdf_path):
                if os.path.isfile(os.path.join(self.pdf_path, f)) and not f.startswith("."):
                    return True
            return False
    def loading(self, pdf_path, filename):
        raw_pdf_elements = partition_pdf(
            filename=pdf_path + filename,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf=True,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure=True,
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            # Chunking params to aggregate text blocks
            # Attempt to create a new chunk 3800 chars
            # Attempt to keep chunks > 2000 chars
            # Hard max on chunks
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=pdf_path,
        )
        return raw_pdf_elements
    def splitting(self, raw_pdf_elements):
        category_counts = {}
        for element in raw_pdf_elements:
            category = str(type(element))
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        # Unique_categories will have unique elements
        unique_categories = set(category_counts.keys())

        categorized_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))

        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]

        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]
        

        img_summaries = []
        file_paths = glob.glob(os.path.expanduser(os.path.join(self.pdf_path, "*.txt")))

        for file_path in file_paths:
            with open(file_path, "r") as file:
                img_summaries.append(file.read())

        return table_elements, text_elements, img_summaries
    
    def embed_documents(self, prompt_text, table_elements, text_elements, img_summaries, prompt):
        summarize_chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
        # Apply to text
        texts = [i.text for i in text_elements]
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        # Apply to tables
        tables = [i.text for i in table_elements]
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        logging_header = "clip_model_load: total allocated memory: 201.27 MB\n\n"
        cleaned_img_summary = [s.split(logging_header, 1)[1].strip() if logging_header in s else s.strip()
        for s in img_summaries]
        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=id_key
        )

        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        if summary_texts:
            retriever.vectorstore.add_documents(summary_texts)
            retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = []
        filtered_table_ids = []
        filtered_tables = []

        for i, s in enumerate(table_summaries):
            if s and s.strip():
                summary_tables.append(Document(page_content=s, metadata={id_key: table_ids[i]}))
                filtered_table_ids.append(table_ids[i])
                filtered_tables.append(tables[i])  # ensure content matches ID

        if summary_tables:  # Only insert if non-empty
            retriever.vectorstore.add_documents(summary_tables)
            retriever.docstore.mset(list(zip(filtered_table_ids, filtered_tables)))


    # Add image summaries
    
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)
        ]
        
        if summary_img:
            retriever.vectorstore.add_documents(summary_img)
            retriever.docstore.mset(list(zip(img_ids, cleaned_img_summary)))
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
            llm_with_tools = self.model.bind_tools([self.make_retrieve(vectorstore)])
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
            "don't know. Do not use the knowledge you extracted on the internet. Focus solely on the retrieved documents only"
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
        response = self.model.invoke(prompt)
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