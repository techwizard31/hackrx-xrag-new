import os
from dotenv import load_dotenv

load_dotenv()


# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from langchain_openai import ChatOpenAI
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from spacy.cli import download
from spacy.lang.en import English
from graphr.helper_functions import *
from graphr.evaluation.evalute_rag import *
from utils.helpers import document_splitter


nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from parser.PDFParser import PDFParser
from utils.helpers import *
from models.graph_rag import GraphRAG


app = FastAPI(title="PDF to Markdown Converter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = PDFParser(api_key=LLAMA_CLOUD_API_KEY)

@app.get("/")
async def root():
    try:
        client.admin.command('ping')
        print("✅ Connected to MongoDB!")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

    db = client["llama_docs"]
    collection = db["parsed_pdfs"]

    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    # Check MongoDB
    existing = collection.find_one({"url": pdf_url})
    if existing:
        print("📄 Parsed result fetched from DB")
        parsed_text = existing["parsed"]
    else:
        # Parse using LlamaParse
        parser = PDFParser(api_key=LLAMA_CLOUD_API_KEY)
        result = parser.parse_pdf(pdf_url)
        parsed_text = [doc.text for doc in result]

        # Save to MongoDB
        collection.insert_one({"url": pdf_url, "parsed": parsed_text})
        print("✅ Parsed and stored in DB")

        # Optionally save to .pkl
        # with open("parsed_result.pkl", "wb") as f:
        #     pickle.dump(result, f)
        # print("💾 Saved parsed result to 'parsed_result.pkl'")

    # If you prefer to read from file (fallback):
    if not existing:
        with open("parsed_result.pkl", "rb") as f:
            result = pickle.load(f)
        print("📂 Loaded parsed result from 'parsed_result.pkl'")

    # === Chunking and GraphRAG Processing ===
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = document_splitter(text_splitter, result)

    graph_rag = GraphRAG()
    graph_rag.process_documents(docs)

    query = "Are there any sub-limits on room rent and ICU charges for Plan A?"
    response = graph_rag.query(query)

    print("=======================+>")
    print(response)

    return JSONResponse(content={"response": response})

# pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# # result = parser.parse_pdf(pdf_url)
# # print("Parsing completed!")

# # ==================================================+>

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # ==================================================+>
# import pickle

# # # Save the parsed result to a file
# # with open('parsed_result.pkl', 'wb') as f:
# #     pickle.dump(result, f)

# # print("Parsed result saved to 'parsed_result.pkl'.")
# # # read the parsed result from the file
# with open('parsed_result.pkl', 'rb') as f:
#     result = pickle.load(f)

# print("Parsed result loaded from 'parsed_result.pkl'.")

# # ==================================================+>

# docs = document_splitter(text_splitter, result)
# # print(docs)

# graph_rag = GraphRAG()

# graph_rag.process_documents(docs)

# query = "Are there any sub-limits on room rent and ICU charges for Plan A?"
# response = graph_rag.query(query)


# print("=======================+>")
# print("\n")
# print(response)
