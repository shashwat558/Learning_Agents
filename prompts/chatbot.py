from langchain_huggingface import HuggingFaceEndpoint
 
import os 
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model = HuggingFaceEndpoint(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="chat",
    model_kwargs={"token": hf_token},



)

chat_history = []


while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if(user_input == "exit"):
        break
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI: ", result)
