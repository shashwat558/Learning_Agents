from langchain_openai import ChatOpenAI
import os
from typing import TypedDict
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN");

model = ChatOpenAI()

class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Today i messed up in my exam but i am looking forward to smashing the next one
""")
print(result)