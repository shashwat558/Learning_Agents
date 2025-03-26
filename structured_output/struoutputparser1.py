from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser;

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
                           
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following:  {text}",
    input_variables=["text"]
)

parser = StrOutputParser();



chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Black hole"})





print(result)