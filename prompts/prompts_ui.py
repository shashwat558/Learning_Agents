from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate, load_prompt
import os
import streamlit as st
from dotenv import load_dotenv

# Load API token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


model = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,  
    temperature=0.7,  
    max_length=512    
)


st.header("Movie Analytics Tool")

genre_input = st.selectbox("Select Movie Genre", ["Thriller", "Comedy", "Feel good", "Drama"])
industry_input = st.selectbox("Select Industry", ["Hollywood", "Bollywood", "Korean"])
time_input = st.selectbox("Select Release Time", ["New", "Old", "Very Old"])


prompt_template = load_prompt("template.json")





if st.button("Analyze"):
    chain = prompt_template | model
    result = chain.invoke({
        "genre_input": genre_input,
        "industry_input": industry_input,
        "time_input": time_input
    })
     
    st.write(result)
