from langchain_core.prompts import PromptTemplate


prompt_template = PromptTemplate(
    template="Generate an analysis of a {genre_input} {industry_input} {time_input} movie. "
             "Discuss its common themes, storytelling style, and audience appeal. "
             "Mention notable films from this category and highlight what makes this genre unique within the {industry_input} industry.",
    input_variables=["genre_input", "industry_input", "time_input"],
    validate_template=True
)


prompt_template.save("template.json")