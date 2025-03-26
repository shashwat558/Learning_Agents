from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema;

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1",description="fact 1 about the topic"),

    ResponseSchema(name="fact_2",description="fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="fact 3 about the topic"),
    ResponseSchema(name="fact_4",description="fact 4 about the topic"),
    ResponseSchema(name="fact_5",description="fact 5 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 5 facts about the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}   
    
)

prompt = template.invoke({"topic": "artificial intelligence"})

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result["fact_1"])
