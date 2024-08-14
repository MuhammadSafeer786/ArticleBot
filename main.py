import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import SeleniumURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
st.title("ArticleBot: Article Research Tool 📈")
st.sidebar.title("Article URLs")

url = st.sidebar.text_input(f"Enter URL here...")

urls = [url]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    main_placeholder.text("URL Loader...Started...✅✅✅")
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text(f"{data}")
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    doc = text_splitter.split_text(data)
    doc = Document(page_content=doc, metadata={"source": f"{url}"})
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(doc, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                # Split the sources by newline
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
