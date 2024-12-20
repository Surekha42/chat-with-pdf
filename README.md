Chat with PDF using RAG Pipeline

This project provides a Streamlit application that allows users to interact with PDF documents by asking questions. It uses the Retrieval-Augmented Generation (RAG) pipeline to extract context from the PDFs and generate detailed answers using Google's Generative AI models.

Features

Upload and process multiple PDF files.

Extract text from PDFs and split it into manageable chunks.

Create and manage a FAISS index for efficient document retrieval.

Handle both natural language and tabular data queries.

Display tabular data in a structured format using Pandas.

Real-time conversational AI integration.

Tech Stack

Python: Core programming language.

Streamlit: Web interface for interaction.

PyPDF2: PDF text extraction.

LangChain: Text splitting and RAG pipeline integration.

FAISS: Vector store for document similarity search.

Google Generative AI: Embeddings and conversational AI models.

Pandas: Tabular data handling.

Getting Started

Prerequisites

Ensure the following are installed:

Python (>= 3.8)

Streamlit

PyPDF2

LangChain

FAISS

Google Generative AI SDK

dotenv

Installation

Clone the repository:

git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file in the project root.

Add your Google API key:

GOOGLE_API_KEY=your_google_api_key_here

Run the application:

streamlit run app.py

Usage

Upload PDFs: Use the sidebar to upload one or more PDF files.

Process PDFs: Click the "Submit & Process" button to extract and index text from the PDFs.

Ask Questions: Use the text input box on the main page to ask questions about the uploaded PDFs.

View Results: Receive answers in text or tabular format, depending on the content.

Folder Structure

chat-with-pdf/
|— app.py                 # Main Streamlit app file
|— requirements.txt       # Python dependencies
|— .env                   # Environment variables (not included in the repo)
|— faiss_index/           # Folder for FAISS index (created during runtime)

Important Notes

Ensure your .env file contains a valid GOOGLE_API_KEY.

FAISS index is stored locally in the faiss_index folder and will be overwritten when new PDFs are processed.

Clear the faiss_index folder if encountering errors or starting fresh.
