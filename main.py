import os
from dotenv import load_dotenv

load_dotenv()
# print(os.getenv('OPENAI_API_KEY'))

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.responses import JSONResponse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage  # Add this import at the top
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

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
uri = os.getenv("MONGO_URI")
# client = MongoClient(uri, server_api=ServerApi('1'))

app = FastAPI(
    title="PDF to Markdown Converter",
    version="1.0.0",
    root_path="/api/v1"
)

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
    # client = MongoClient(uri, server_api=ServerApi('1'))

    # try:
    #     client.admin.command('ping')
    #     print("✅ Connected to MongoDB!")
    # except Exception as e:
    #     print(e)
    #     raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

    # db = client["llama_docs"]
    # collection = db["parsed_pdfs"]

    # pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    # # Check MongoDB
    # existing = collection.find_one({"url": pdf_url})
    # if existing:
    #     print("📄 Parsed result fetched from DB")
    #     parsed_text = existing["parsed"]
    # else:
    #     # Parse using LlamaParse
    #     parser = PDFParser(api_key=LLAMA_CLOUD_API_KEY)
    #     result = parser.parse_pdf(pdf_url)
    #     parsed_text = [doc.text for doc in result]

    #     # Save to MongoDB
    #     collection.insert_one({"url": pdf_url, "parsed": parsed_text})
    #     print("✅ Parsed and stored in DB")

    #     # Optionally save to .pkl
    #     # with open("parsed_result.pkl", "wb") as f:
    #     #     pickle.dump(result, f)
    #     # print("💾 Saved parsed result to 'parsed_result.pkl'")

    # # If you prefer to read from file (fallback):
    import pickle
    # if not existing:
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


from pydantic import BaseModel
from typing import List

# Add these request/response models
class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackrxResponse(BaseModel):
    answers: List[str]

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

VALID_API_KEY = os.getenv("API_SECRET_KEY")


def verify_bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    
    if token != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

@app.post("/hackrx/run")
async def hackrx_run(request: HackrxRequest, token: str = Depends(verify_bearer_token)):
    try:
        # Extract PDF URL and questions from request
        pdf_url = request.documents
        questions = request.questions

        print(f"Processing PDF: {pdf_url}")
        print(f"Number of questions: {len(questions)}")

        print("🔄 Parsing PDF...")
        parser_instance = PDFParser(api_key=LLAMA_CLOUD_API_KEY)
        result = parser_instance.parse_pdf(pdf_url)

        # === Chunking and GraphRAG Processing ===
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = document_splitter(text_splitter, result)

        # Initialize GraphRAG
        graph_rag = GraphRAG()
        graph_rag.process_documents(docs)
        print("✅ GraphRAG processing completed")

        # Function to process a single question
        def process_single_question(question_data):
            index, question = question_data
            try:
                print(f"🔍 Processing question {index + 1}: {question[:50]}...")
                response = graph_rag.query(question)

                # Handle AIMessage objects specifically
                if isinstance(response, AIMessage):
                    answer_text = response.content
                elif isinstance(response, str):
                    answer_text = response
                else:
                    answer_text = str(response)

                print(f"✅ Question {index + 1} answered")
                return (index, answer_text)
            except Exception as e:
                print(f"❌ Error processing question {index + 1}: {str(e)}")
                return (index, f"Error processing question: {str(e)}")

        # Process all questions concurrently
        print("🚀 Starting concurrent processing of all questions...")

        # Use ThreadPoolExecutor to run queries concurrently
        with ThreadPoolExecutor(max_workers=min(len(questions), 10)) as executor:
            # Submit all questions with their indices
            question_data = list(enumerate(questions))
            future_to_question = {
                executor.submit(process_single_question, q_data): q_data
                for q_data in question_data
            }

            # Collect results with their original indices
            results = []
            for future in future_to_question:
                try:
                    index, answer = future.result()
                    results.append((index, answer))
                except Exception as e:
                    question_index = future_to_question[future][0]
                    results.append((question_index, f"Error processing question: {str(e)}"))

        # Sort results by original index to maintain order
        results.sort(key=lambda x: x[0])
        answers = [answer for _, answer in results]

        print("✅ All questions processed successfully")

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        print(f"❌ Error in hackrx_run: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
