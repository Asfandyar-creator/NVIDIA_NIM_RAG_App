import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

# Loading the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')



def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./New folder')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


    

st.title('NVIDIA NIM PDF Q&A')
llm = ChatNVIDIA(model = 'meta/llama3-70b-instruct')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    
    """
)

prompt1 = st.text_input("Enter your question from Documents.")

if st.button("Documents Embeddings"):
    vector_embeddings()
    st.write("Vector store DB is ready!")



if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrival_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrival_chain.invoke({'input': prompt1})
    print('Response time: ', time.process_time() - start)
    st.write(response['answer'])



    # # Write a streamlit expander
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant chunk
    #     for i, doc in enumerate(response['answer']):
    #         st.write(doc.page_content)
    #         st.write('------------------------------------')