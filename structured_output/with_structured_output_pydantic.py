
from langchain_openai import ChatOpenAI
import os
from pydantic import Field
from typing import TypedDict, Annotated, Optional
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN");

model = ChatOpenAI()

class Review(TypedDict):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in a list")
    summary: str = Field(description="A brief summary of a review")
    sentiments: str = Field(description="Return sentiment of the review either negative, positive or neutral")

    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Today i messed up in my exam but i am looking forward to smashing the next one
""")
print(result)