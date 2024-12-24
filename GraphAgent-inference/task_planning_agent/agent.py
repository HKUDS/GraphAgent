from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
# from utils.langgraph_utils import create_langgraph_agent
from utils.storage import mem_store
from .prompts import few_shot_example_1, few_shot_example_2, few_shot_example_3
from langchain.prompts import SystemMessagePromptTemplate
from utils.llm import llm

system_prompt_template = """
    You are very powerful assistant developed by HKU Data Intelligance Lab for various graph-related tasks from diverse user inputs. 
    You can do great in parsing the following important properties from the user input:

    1. "knowledge_text". Full and comprehensive user input text that contains inherent knowledge, excluding any tasks, questions or labels information.
    2. "task_type". Type of the task to handle, must be one of {task_enum}. You should infer the graph task to handle from the user input.
    3. "user_annotation". Any additional information provided by the user, such as task description, label candidates for predictive tasks, or and generation requirements for generative tasks, etc.

    You must follow the format instructions to provide the output in the correct format, which can be parsed into JSON.
    You must use the correct key names in the output json object. An example for output format is provided below:
    {{
        "knowledge_text": "",
        "task_type": "",
        "user_annotation": ""
    }}

    Here are realistic input-output examples for you to better understand the requirements:
    {few_shot_examples}
    """

TaskEnum = {
    "predictive",
    "generative"
}

class TaskFormulationOutput(BaseModel):
    knowledge_text: str = Field(description="Full and comprehensive user input text that contains inherent knowledge, excluding any tasks, questions or labels information.")
    task_type: str = Field(description=f"Type of the task to handle. Must be one of {TaskEnum}")
    user_annotation: str = Field(description="Any additional information provided by the user, such as task description, label candidates for predictive tasks, or and generation requirements for generative tasks, etc.")

output_parser = JsonOutputParser(pydantic_object=TaskFormulationOutput)

system_prompt = PromptTemplate(
    template=system_prompt_template,
    input_variables=[],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions(),
        "task_enum": ",".join(TaskEnum),
        "few_shot_examples": few_shot_example_1 + few_shot_example_2 + few_shot_example_3,
    },
)
prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=system_prompt), ("user", "{input}")])

def save_output_to_mem_store(output):
    for key, value in output.items():
        mem_store.mset([(key, value)])
    
    return output

proto_agent = llm | output_parser | save_output_to_mem_store

agent = prompt | llm | output_parser | save_output_to_mem_store