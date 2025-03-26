from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser;

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me the cast, director and main character name of a movie {name} \n {format_instruction}",
    input_variables=["name"],
    partial_variables= {"format_instruction": parser.get_format_instructions()}
)

# prompt = template1.format()

# result = model.invoke(prompt);

# json_result = parser.parse(result.content)

chain = template1 | model | parser
result = chain.invoke({"name": "Django unchained"})
print(result["cast"])