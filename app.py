import os
import Flask
import dotenv
from dotenv import load_dotenv
from flask import render_template, jsonify, request
# Load environment variables FIRST
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app early
app = flask(__name__)

# API Key Validation
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Check your .env file.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

# --- Rest of Imports ---

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Routes Must Be Defined Before App Run ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    result = rag_chain.invoke({"input": msg})
    return str(result["answer"])

# --- Application Setup ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.35}
)

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    max_output_tokens=500
)

system_prompt = """
You MUST follow these rules:
1. Answer ONLY using the context below
2. If context says "NO_RELEVANT_CONTEXT_FOUND", respond "I don't know"
3. Never mention you're an AI assistant
4. NEVER USE YOUR OWN KNOWLEDGE

You are a medical expert assistant. Provide detailed explanations using the context below.
Include:
1. Key definitions
2. Common symptoms (if applicable)
3. Treatment options
4. Prevention tips

If context is insufficient, say "I don't have enough medical information about that."

CONTEXT: {context}
QUESTION: {input}

Answer in 5-7 sentences using professional medical terminology:

Context: {context}

IF YOU THINK THAT SUFFICIENT KNOWLEDGE IS NOT THERE IN PINECONE DATABASE  SAY "NO_RELEVANT_CONTEXT_FOUND"
"""  # Keep your existing prompt here

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)

    



