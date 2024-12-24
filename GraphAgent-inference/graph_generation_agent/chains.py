from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langchain.prompts import SystemMessagePromptTemplate
from .scaffold_text_and_keywords_prompts import scaffold_texts_from_input_example, keywords_from_scaffold_text_example
from .scaffold_node_extraction_prompts import few_shot_example_1, few_shot_example_2, few_shot_example_3

class ScaffoldNodeExtractionOutput(BaseModel):
    thinking: str = Field(description="Your brief thoughts in a structured manner on what are the scaffold nodes.")
    scaffold_nodes: list[dict] = Field(description="A list of scaffold nodes.")

def create_scaffold_node_extraction_chain(llm, few_shot_examples = []):
    if len(few_shot_examples) == 0:
        few_shot_examples = ["\n".join(few_shot_example_1), "\n".join(few_shot_example_2), "\n".join(few_shot_example_3)]

    system_prompt_template = """
    You are very powerful assistant developed by HKU Data Intelligance Lab for various graph-related tasks from diverse user inputs.
    Given a user input that contains two parts:
    1. "knowledge_text". Full and comprehensive user input text that contains inherent knowledge and rich implicit semantic relationships and key concepts.
    2. "user_annotation". Additional information provided by the user, such as task description, label candidates for predictive tasks, or and generation requirements for generative tasks, etc.

    Your task is to generate a list of abstract and top-high-level informative concepts, as which we call "scaffold nodes", given the above two parts.
    Scaffold nodes are:
    - abstract and high-level, representing the top-level concepts or entities in the text. For example, given a complex movie script, the scaffold nodes could be "plot", "character", "dialogue", "setting", "theme", etc.
    - automatic and non-redundant. Each scaffold node should be corresponded to a certain aspect, key topic or entity in the text.
    - task-specific. For example, if the task is to summarize the input story, the scaffold nodes could be "main characters", "background", "temporal structure", "thematic threads", etc. However, if the task is regarding specific characters, the scaffold nodes could be "main character", "supporting character", "character development", etc.

    You are required to return the following two parts in the JSON format:

    1. "thinking". You should first think step by step about the task and the input, and then provide your thoughts in a structured manner on what are the scaffold nodes.
    2. "scaffold_nodes". A list of scaffold nodes.

    You must follow the format instructions to provide the output in the correct format, which can be parsed into JSON.
    You must use the correct key names in the output json object. An example for output format is provided below:
    {{
        "thinking": "",
        "scaffold_nodes": []
    }}

    Here are realistic input-output examples for you to better understand the task:
    {few_shot_examples}
    Directly return the json dictionary, starting with \"{{\".
    """

    output_parser = JsonOutputParser(pydantic_object=ScaffoldNodeExtractionOutput)

    system_prompt = PromptTemplate(
        template=system_prompt_template,
        input_variables=[],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions(),
            "few_shot_examples": "\n\n".join(few_shot_examples),
        },
    )
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=system_prompt), ("user", "Knowledge Text: {knowledge_text}\nUser Annotation: {user_annotation}")])

    chain = (
        {
            "knowledge_text": lambda x: x["knowledge_text"],
            "user_annotation": lambda x: x["user_annotation"],
        }
        | prompt
        | llm
        # | RunnableLambda(lambda x: (mem_store.mset([("scaffold_node_extraction_output", x)]), x))
        | output_parser
    )
    return chain

class ScaffoldTextParsingOutput(BaseModel):
    scaffold_texts: list[dict] = Field(description="a list of parsed texts for each scaffold node")
    
def create_scaffold_text_parsing_chain(llm, few_shot_examples = []):
    if len(few_shot_examples) == 0:
        few_shot_examples = ["\n".join(scaffold_texts_from_input_example)]

    system_prompt = """
        You are a powerful assistant in parsing corresponding texts in the user input given a list of scaffold nodes. Each scaffold node represents a high-level key point or topic in the text, and your goal is to provide comprehensive and detailed texts related to each scaffold node. The texts should be parsed from the given input. Texts should be detailed and you should never miss any important information.

        You can never miss any node in the input.
        You should parse corresponding texts for each scaffold node in the input. You should always return the same number of scaffold nodes as the input.

        Return the parsed texts for each scaffold node in the JSON format.

        You are given realistic input-output examples to assist you in the task: {few_shot_examples}
    """
    system_prompt = PromptTemplate(
        template=system_prompt,
        input_variables=[],
        # partial_variables={"one_shot_example": peer_review_decision_scaffold_texts_from_input_example}
        partial_variables={"few_shot_examples": "\n\n".join(few_shot_examples)}
    )
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=system_prompt), ("user", "Knowledge Text: {knowledge_text}\nScaffold Nodes: {scaffold_nodes}")])
    output_parser = JsonOutputParser(pydantic_object=ScaffoldTextParsingOutput)
    chain = (
        {
            "knowledge_text": lambda x: x["knowledge_text"],
            "scaffold_nodes": lambda x: x["scaffold_nodes_extraction_output"]["scaffold_nodes"],
        }
        | prompt
        | llm
        # | RunnableLambda(lambda x: (mem_store.mset([("scaffold_text_parsing_output", x)]), x))
        | output_parser
    )
    return chain


class KeywordsOutput(BaseModel):
    keywords: list[dict] = Field(description="a list of extracted keywords for each scaffold node")

def create_keywords_from_scaffold_text_chain(llm, few_shot_examples = []):
    if len(few_shot_examples) == 0:
        few_shot_examples = ["\n".join(keywords_from_scaffold_text_example)]

    system_prompt = """
        You are a powerful assistant in extracting keywords from the parsed texts of scaffold nodes. Keywords should be informative and representative of the key points in the text.

        You also need to provide a description of the extracted keywords for each scaffold node. The description should be detailed and informative, and strictly contains two parts: 1) a brief description of the keywords based on the contexts in the text, and 2) a detailed description of the keywords based on your own knowledge.

        Return the extracted keywords for each scaffold node in the JSON format.

        You are given an example to assist you in the task: {few_shot_examples}. Directly return the json dictionary, starting with \"{{\".
    """
    system_prompt = PromptTemplate(
        template=system_prompt,
        input_variables=[],
        partial_variables={"few_shot_examples": "\n\n".join(few_shot_examples)}
    )
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=system_prompt), ("user", "input: {input}")])
    output_parser = JsonOutputParser(pydantic_object=KeywordsOutput)
    chain = (
        prompt
        | llm
        # | RunnableLambda(lambda x: (mem_store.mset([("keywords_from_scaffold_text_output", x)]), x))
        | output_parser
    )
    return chain