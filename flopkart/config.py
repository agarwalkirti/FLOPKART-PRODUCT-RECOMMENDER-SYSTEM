import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ASTRA_DB_API_ENDPOINT = os.getenv("FLOPKART_ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("FLOPKART_ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("FLOPKART_ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RAG_MODEL = "llama-3.1-8b-instant"