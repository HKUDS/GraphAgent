from torch_geometric.data import HeteroData
from utils.llm import llm
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from utils.formatter import CompactedStr
import json

def create_node_edge_type_description_chain():
    system_prompt_template = CompactedStr("""
        You are an extremely prowerful assistant in providing rich descriptions for node and edge types on a specific graph.
        You are expected to perform well in the following task:
        Give a set of node types and edge types on a specific graph, you should generate rich descriptions for each node and edge type.
        For example:
            node types: ["paper", "research_background", "research_question", "methodology", "key_results", "keyword"]
            edge types: ["paper, has_property, research_background", "paper, has_property, research_question", "paper, has_property, methodology", "paper, has_property, key_results", "paper, has_keyword, keyword"]

        Please provide rich descriptions for each node and edge type as:

            "node_types": {{
                    "paper": "This node type represents a research paper",
                    "research_background": "This node type represents the background of the research",
                    "research_question": "This node type represents the research question",
                    "methodology": "This node type represents the methodology used in the research",
                    "key_results": "This node type represents the key results of the research",
                    "keyword": "This node type represents the keyword of the research"
                }}

            "edge_types": {{
                    "paper, has_property, research_background": "This edge type represents the relationship between a paper and its background",
                    "paper, has_property, research_question": "This edge type represents the relationship between a paper and its research question",
                    "paper, has_property, methodology": "This edge type represents the relationship between a paper and its methodology",
                    "paper, has_property, key_results": "This edge type represents the relationship between a paper and its key results",
                    "paper, has_keyword, keyword": "This edge type represents the relationship between a paper and its keyword"
                }}

        Strictly follow the format instructions to provide the output in the correct format, which can be parsed into JSON.
    """).apply_formatting()

    class NodeAndEdgeTypesOutput(BaseModel):
        node_types: dict
        edge_types: dict

    output_parser = JsonOutputParser(pydantic_object=NodeAndEdgeTypesOutput)

    system_prompt = PromptTemplate(
        template=system_prompt_template,
        input_variables=[],
        # partial_variables={
        #     "format_instructions": output_parser.get_format_instructions(),
        # },
    )
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=system_prompt), ("user", "node types: {node_type_list}\nedge types: {edge_type_list}")])

    node_edge_type_description_chain = prompt | (lambda x: (print("prompt: ", x), x)[1]) | llm | output_parser | (lambda x: (print("output: ", x), x)[1])

    return node_edge_type_description_chain

node_edge_type_description_chain = create_node_edge_type_description_chain()

def node_edge_type_emb_from_sbert(pyg_graph: HeteroData) -> Path :
    node_type_list = list(pyg_graph.x_dict.keys())
    edge_type_list = list(pyg_graph.edge_index_dict.keys())

    natural_langauge_desc = node_edge_type_description_chain.invoke({"node_type_list": node_type_list, "edge_type_list": edge_type_list})
    # natural_langauge_desc = json.loads(natural_langauge_desc)
    node_type_desc = natural_langauge_desc["node_types"]
    edge_type_desc = natural_langauge_desc["edge_types"]


    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    node_type_descriptions = list(node_type_desc.values())
    edge_type_descriptions = list(edge_type_desc.values())

    node_type_embs = model.encode(node_type_descriptions, convert_to_tensor=True).cpu()
    edge_type_embs = model.encode(edge_type_descriptions, convert_to_tensor=True).cpu()

    node_type_emb_dict = dict(zip(node_type_desc.keys(), node_type_embs))
    edge_type_emb_dict = {tuple(k1.strip() for k1 in k.split(",")): v for k, v in zip(edge_type_desc.keys(), edge_type_embs)}

    save_path = Path(f"tmp/node_edge_type_emb_dict.tmp.{pyg_graph.graph_id}.pt")

    torch.save({
        "node_type_dict": node_type_desc,
        "edge_type_dict": edge_type_desc,
        "node_type_emb_dict": node_type_emb_dict,
        "edge_type_emb_dict": edge_type_emb_dict,
    }, save_path)

    return save_path

def node_feat_from_sbert(pyg_graph: HeteroData) -> HeteroData:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    node_type_dict = pyg_graph.x_dict
    for node_type in node_type_dict:
        embs = model.encode(pyg_graph[node_type].description, convert_to_tensor=True).cpu()
        pyg_graph[node_type].x = embs
    
    return pyg_graph
    
