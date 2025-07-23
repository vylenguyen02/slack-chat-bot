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


class TextHandlingService:
    def splitting(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
        all_splits = text_splitter.split_text(docs)
        return all_splits

    def embed_documents(self, all_splits, embedder, collection, search_index):
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
                    texts=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
        ids=[str(val) for val in range(len(all_splits))],
        vectorstore.create_vector_search_index(dimensions=3072)
        vectorstore.add_texts(all_splits,ids=ids[0])
    
    
    