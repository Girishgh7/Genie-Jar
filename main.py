import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

#Sidebar:
with st.sidebar:
    st.title('🧞 ☁️ Genie Jar')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Hugging_face](https://huggingface.co/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(7)
    st.write('Made with 🪄of genie and llms ❤️')
    st.write('Made by 🐒Girish_GH🤖')


def main():
    st.header("Genie JAR🧞🪄")
    load_dotenv('.env')
    #upload pdf
    pdf=st.file_uploader("Upload your PDF",type='pdf')
    if pdf is not None:
     pdf_reader=PdfReader(pdf)
     text=''
     for page in pdf_reader.pages:
         text+=page.extract_text()
     #st.write(pdf_reader)
     text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=1000,
         chunk_overlap=200,
         length_function=len
         )
     chunks=text_splitter.split_text(text=text)
     store_name = pdf.name[:-4]
     st.write(f'{store_name}')
    if os.path.exists(f'{store_name}.pkl'):
        with open(f'{store_name}.pkl','rb') as f:
            VectorStore=pickle.load(f)
        #st.write('Embeddings loaded from the Disk')
    else:    
        #embeddings
     embeddings=OpenAIEmbeddings()
     VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
     with open(f"{store_name}.pkl",'wb') as f:
          pickle.dump(VectorStore,f)
    #accept user query
    query = st.text_input("Ask questions about your PDF file to Genie:")
    if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(temperature=0.7,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__=='__main__':
    main()
