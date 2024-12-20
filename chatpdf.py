import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil  # To clear the FAISS index folder
import pandas as pd  # To handle tabular data

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=api_key)


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to generate and save vector store
def get_vector_store(text_chunks):
    # Clear any existing FAISS index
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    
    # Create new embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to load the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context" — don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to check if the text contains tabular data
def is_tabular_data(text):
    """
    Check if the text contains tabular data by identifying patterns like rows and columns.
    This function can be improved based on the format of the tabular data.
    """
    return "\n" in text and any(delimiter in text for delimiter in [",", "\t", "|"])


# Function to parse tabular data into a Pandas DataFrame
def parse_table(text):
    """
    Parse markdown-style tabular data from the response text into a Pandas DataFrame.
    Assumes the table is formatted with '|' delimiters and has a header row.
    """
    # Extract lines that contain the table
    lines = [line.strip() for line in text.splitlines() if "|" in line]
    
    # Remove separator lines (like `|---|---|---|`)
    table_lines = [line for line in lines if not all(c in "-| " for c in line)]
    
    # Split each line into columns
    rows = [line.split("|")[1:-1] for line in table_lines]  # Exclude leading and trailing '|'
    
    # Remove extra whitespace and ensure consistent column count
    rows = [list(map(str.strip, row)) for row in rows]
    
    # Convert to DataFrame
    if len(rows) < 2:
        raise ValueError("Insufficient data for table parsing.")
    headers = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=headers)


# Function to handle user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if FAISS index exists
    if not os.path.exists("faiss_index"):
        st.error("No FAISS index found. Please upload and process PDF files first.")
        return
    
    try:
        # Allow safe deserialization since the index is trusted
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    # Perform similarity search and get the response
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    output_text = response['output_text']

    # Check if the output contains tabular data
    if is_tabular_data(output_text):
        try:
            # Convert the tabular data from text to a DataFrame
            df = parse_table(output_text)
            st.table(df)  # Display the table in Streamlit
        except Exception as e:
            st.error(f"Error parsing table: {e}")
            st.markdown(f"Reply: {output_text}")
    else:
        st.markdown(f"Reply: {output_text}")


# Streamlit app main function
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using RAG pipeline")

    # User input for questions
    user_question = st.text_input("Ask a Question from the PDF Files:")

    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully. You can now ask questions!")


if __name__ == "__main__":
    main()
