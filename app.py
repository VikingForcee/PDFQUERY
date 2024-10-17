import streamlit as st
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import numpy as np
from langchain.vectorstores import FAISS
from chromadb import Client
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please ensure it's set in the environment variables.")
else:
    genai.configure(api_key=api_key)

# Initialize Chroma client
persist_directory = "chroma_db"
chroma_client = Client(Settings(
    persist_directory=persist_directory,
    anonymized_telemetry=False
))

# Global dictionary to store text chunks
text_chunks_dict = {}

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Store embeddings in Chroma and text chunks in dictionary
def store_in_chroma(text_chunks):
    global text_chunks_dict
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create or get a collection in Chroma
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

    # Store each embedding in Chroma and text chunk in dictionary
    for i, chunk in enumerate(text_chunks):
        embedding_vector = embeddings.embed_query(chunk)
        doc_id = f"doc_{i}"
        collection.add(
            embeddings=[embedding_vector],
            ids=[doc_id],
            metadatas=[{"text_chunk": chunk[:100] + "..."}]  # Store a preview of the text chunk in metadata
        )
        text_chunks_dict[doc_id] = chunk

# Set up the conversation chain with a template
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Use the provided context to answer the question as thoroughly as possible. 
    If the answer is not in the provided context, use your knowledge and reasoning abilities to provide an answer. 
    If you are uncertain, say, "Based on what I know, ..." and provide a well-informed guess or reasoning.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Handle user input
# Handle user input
def user_input(user_question):
    try:
        # Set embeddings using Google Generative AI
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load FAISS index with dangerous deserialization allowed
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search for relevant documents
        docs = new_db.similarity_search(user_question)
        
        # Get conversation chain and generate a response
        chain = get_conversational_chain()
        if docs:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response.get("output_text", "No output text found."))
        else:
            st.write("No relevant documents found. Using generative model to answer the question.")
            # Fallback to generative reasoning if no docs are found
            generative_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
            prompt_template = """
            Answer the following question using your own knowledge and reasoning. If the context is available, use it; 
            if not, provide your own understanding. Always give a thoughtful and detailed answer.

            Question: \n{question}\n
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
            gen_chain = load_qa_chain(generative_model, chain_type="stuff", prompt=prompt)
            gen_response = gen_chain.run({"question": user_question})
            st.write("Reply: ", gen_response)
    
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        logging.exception("Error processing user input.")

# Function to view Chroma database contents
def view_chroma_database():
    try:
        collection = chroma_client.get_collection(name="pdf_embeddings")
        count = collection.count()
        st.write(f"Total embeddings in the database: {count}")
        
        # Get all embeddings (use with caution if you have a large number of embeddings)
        results = collection.get(include=['embeddings', 'metadatas'])
        
        for i, (embedding, metadata) in enumerate(zip(results['embeddings'], results['metadatas'])):
            st.write(f"Embedding {i+1}:")
            st.write(f"Embedding (first 10 dimensions): {np.array(embedding[:10])}...")
            st.write(f"Metadata: {metadata}")
            st.write(f"Corresponding text chunk preview: {metadata.get('text_chunk', 'Not found')}")
            st.write("---")
    
    except Exception as e:
        st.error(f"Error viewing Chroma database: {str(e)}")
        logging.exception("Error viewing Chroma database.")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if "pdf_embeddings" not in [col.name for col in chroma_client.list_collections()]:
            st.error("No documents have been processed yet. Please upload and process PDFs first.")
        else:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Extract text from PDFs and store embeddings in Chroma
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        store_in_chroma(text_chunks)
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        logging.exception("Error during PDF processing.")
        
        if st.button("View Chroma Database"):
            view_chroma_database()

if __name__ == "__main__":
    main()