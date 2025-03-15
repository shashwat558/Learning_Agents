from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import os
import streamlit as st


from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
   
    model_kwargs={"token": hf_token} 
)


paper_input = st.selectbox( "Select Movie genre", ["Thriller", "Comedy", "Feel good","drama"] )

style_input = st.selectbox( "Select Industry", ["Hollywood", "Bollywood", "Korean"] ) 

length_input = st.selectbox( "Select release time", ["New", "Old", "very old"] )

template = PromptTemplate(
    """
Generate an analysis of a {length_input} {style_input} {paper_input} movie. Discuss its common themes, storytelling style, and audience appeal. Mention notable films from this category and highlight what makes this genre unique within the {style_input} industry."
"""
)

if st.button("Analayse"):
    result  = model.invoke(user_input)
    print(result)
    st.write(result)