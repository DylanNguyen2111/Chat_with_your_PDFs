from scrape_job_news import get_text_from_top_articles
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import openai
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_pdf_text_from_folder(pdf_docs_folder):
    text = ""
    for filename in os.listdir(pdf_docs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_docs_folder, filename)
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever =  vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain, retriever

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            reference = st.session_state.retriever.get_relevant_documents(user_question)
            print_reference(reference)

def print_reference(reference_documents):
    st.write("Reference text in your PDFs document:")
    for document in reference_documents:
        page_content = document.page_content
        st.write(page_content)

def main():
    # Enable python to access environment variable from .env
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.header("Chat with multiple PDFs :books:")
    user_questions = st.chat_input("Ask a question about your documents:")

    if user_questions:
        handle_userinput(user_questions)

    # Side bar for user to upload their pdf documents
    with st.sidebar:
        st.subheader("Your douments")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Upload'", accept_multiple_files=True)
        # When user upload the button
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation, st.session_state.retriever = get_conversation_chain(vector_store)
        elif st.button("Latest SG Job News"):
            with st.spinner("Processing"):

                raw_text = get_text_from_top_articles("https://www.straitstimes.com/singapore/jobs", 3)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chainp
                st.session_state.conversation, st.session_state.retriever = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()