import os
from langchain_huggingface import HuggingFaceEndpoint


from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
   
    model_kwargs={"token": hf_token} 
)

response = llm.invoke("What is the capital of India?")
print(response)
