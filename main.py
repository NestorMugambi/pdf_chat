import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import openai

st.set_page_config(
    page_title="pdf_assistant",
    page_icon="📘"
)
with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
                    
        
        ''')
        openai_api_key = st.text_input("Enter your OpenAI API Key", key="langchain_search_api_key_openai", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        add_vertical_space(5)
        


load_dotenv()
def main():
    st.header("Chat with your pdf📃📃")
    
    pdf = st.file_uploader("Upload your file" , type ="pdf")
    if pdf:
        if not openai_api_key:
            st.warning("Please add your OpenAI API key to continue.")
            st.stop()        

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text =""

        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text = text)
        #embeddings
       
        store_name = pdf.name[:-4]

        
        if os.path.exists(f"{store_name}"):
            VectorStore = FAISS.load_local(f"{store_name}", OpenAIEmbeddings(),allow_dangerous_deserialization=True)

        else:            
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            
            VectorStore.save_local(f"{store_name}")

    if "messages" not in st.session_state:
       st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    query = st.chat_input("Ask questions about your pdf")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
          st.markdown(query)
        if not openai_api_key:
            st.warning("Please add your OpenAI API key to continue.")
            st.stop()
        docs = VectorStore.similarity_search(query=query)
        llm=OpenAI(openai_api_key=openai_api_key)       

        chain =load_qa_chain(llm=llm,chain_type="stuff")
        response = chain.run(input_documents=docs,question=query)
        with st.chat_message("assistant"):
           st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
       
            

      
if __name__ == "__main__":
     main()
    

   

        
