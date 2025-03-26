from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a 5 line summary on {topic}",
    input_variables=["topic"]
                           
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following:  {topic}",
    input_variables=["topic"]
)

prompt1 = template1.invoke({"topic": "interstellar"})


result = model.invoke(prompt1)

prompt2 = template2.invoke({"topic": "Adolf hitler"})

result2 = model.invoke(prompt2)

print(result, "                                                                                                                         ",result2)