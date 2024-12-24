from utils.storage import mem_store
from utils.llm import llm
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .chains import create_keywords_from_scaffold_text_chain, create_scaffold_text_parsing_chain, create_scaffold_node_extraction_chain
from functools import partial
import json
from .pyg_utils import build_hetero_graph_from_scaffold_keywords_v2
from colorama import Fore, Style

scaffold_node_extraction_chain = create_scaffold_node_extraction_chain(llm)
scaffold_text_parsing_chain = create_scaffold_text_parsing_chain(llm)
keywords_from_scaffold_text_chain = create_keywords_from_scaffold_text_chain(llm)

def save_pyg_graph_to_mem(x):
    mem_store.mset([("pyg_graph", x["pyg_graph"])])
    return x

def parse_keywords_from_scaffold_texts(x, chain=keywords_from_scaffold_text_chain):
    scaffold_texts = x["scaffold_texts_parsing_output"]["scaffold_texts"]
    keywords = {}
    for item in scaffold_texts:
        keywords[item["node_id"]] = chain.invoke({"input": item["text"]})["keywords"]
    return keywords

def print_lambda(message, color=Fore.RESET):
    return lambda x: (print(color + message + json.dumps(x, indent=4) + Style.RESET_ALL), x)[1]

agent = (
    RunnablePassthrough.assign(scaffold_nodes_extraction_output=scaffold_node_extraction_chain)
    | print_lambda("SCAFFOLD NODES DISCOVERY: ", Fore.CYAN)
    | RunnablePassthrough.assign(scaffold_texts_parsing_output=scaffold_text_parsing_chain)
    | print_lambda("KNOWLEDGE AUGMENTATION: ", Fore.GREEN)
    | RunnablePassthrough.assign(keywords=partial(parse_keywords_from_scaffold_texts, chain=keywords_from_scaffold_text_chain))
    | print_lambda("SCAFFOLD NODES DISCOVERY-2 (KEYWORDS): ", Fore.MAGENTA)
    | print_lambda("GRAPH GROUNDING... ", Fore.YELLOW)
    | RunnablePassthrough.assign(pyg_graph=build_hetero_graph_from_scaffold_keywords_v2)
    | RunnableLambda(save_pyg_graph_to_mem)
    | RunnableLambda(lambda x: f"I have constructed a heterogeneous graph based on your request: {x.get('pyg_graph')}")
)

# agent = (
#     # scaffold_node_extraction_chain
#     RunnablePassthrough.assign(scaffold_nodes_extraction_output=scaffold_node_extraction_chain)
#     | RunnableLambda(lambda x: (print("SCAFFOLD NODES DISCOVERY: ", json.dumps(x, indent=4)), x)[1])
#     | RunnablePassthrough.assign(scaffold_texts_parsing_output=scaffold_text_parsing_chain)
#     | RunnableLambda(lambda x: (print("KNOWLEDGE AUGMENTATION: ", json.dumps(x, indent=4)), x)[1])
#     | RunnablePassthrough.assign(keywords=partial(parse_keywords_from_scaffold_texts, chain=keywords_from_scaffold_text_chain))
#     | RunnableLambda(lambda x: (print("SCAFFOLD NODES DISCOVERY-2 (KEYWORDS): ", json.dumps(x, indent=4)), x)[1])
#     | RunnableLambda(lambda x: (print(Fore.YELLOW + "GRAPH GROUNDING... " + Style.RESET_ALL, x), x)[1])
#     | RunnablePassthrough.assign(pyg_graph=build_hetero_graph_from_scaffold_keywords_v2)
#     | RunnableLambda(save_pyg_graph_to_mem)
#     # | RunnableLambda(lambda x: {"output": x})
#     | RunnableLambda(lambda x: f"I have constructed a heterogeneous graph based on your request: {x.get('pyg_graph')}")
# )


if __name__ == "__main__":
    pass