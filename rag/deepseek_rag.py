import streamlit as st  
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import ollama
from langchain.chains.llm import LLMChain
import streamlit as st
from  langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


st.title("This is a RAG system built with deepseek")


upload_file = st.file_uploader("Upload a pdf file", type="pdf");


if upload_file is not None:
    with open("temp.pdf", "wb")as f:
        f.write(upload_file.getvalue())
        
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load();
    
    
    text_splitter = SemanticChunker(HuggingFaceEmbeddings)
    splitted_docs = text_splitter.split_documents(docs)
    
    embedder = HuggingFaceEmbeddings()
    
    vector =  FAISS.from_documents(splitted_docs, embedder)
    retriever = vector.as_retriever(search_type="similarity", kwargs={"k": 3})
    
    llm = ollama(model="deepseek-r1:1.5b")
    
    prompt_chain = PromptTemplate(
     template="""
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:
    """,
    input_variables=["context", "question"]
    )
    
    
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_chain,
        callbacks=None,
        verbose=True
    )
    
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}"
    )
    
    
    combined_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None
    )
    
    qa = create_retrieval_chain(
        combine_documents_chain=combined_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    
    )


user_input = st.text_input("Ask a question related to the PDF :")
if user_input:
        with st.spinner("Processing..."):
            response = qa(user_input)["result"]
            st.write("Response:")
            st.write(response)
else:
    st.write("Please upload a PDF file to proceed.")
  