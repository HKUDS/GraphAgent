from copy import deepcopy
from utils.formatter import CompactedStr

format_template = {
        "graph_id": "",
        "conversations": [
            {
                "from": "human",
                "value": "Given a heterogeneous academic network graph about a {central_node_type}, there are {num_node_types} types of nodes, namely: {node_type_list}. The relationships (meta paths) between different nodes include: {edge_type_list}. By performing automatic graph building on the concepts and keywords of the {central_node_type}, a heterogeneous subgraph is obtained. In the subgraph, \"{central_node_type}\" nodes: <graph>, where the 0-th node is the central node that represents a {central_node_type} with the following information: \n{graph_raw_text}  \n{other_node_type_and_graph_tokens}. \nQuestion: {user_input} "
            },
            {
                "from": "gpt",
                "value": ""
            }
        ]
    }

def construct_higpt_format_instruction(pyg_graph, graph_raw_text, user_input, few_shot_prompt=None):
    node_type_list = list(pyg_graph.x_dict.keys())
    edge_type_list = list(pyg_graph.edge_index_dict.keys())
    # import pdb; pdb.set_trace()
    node_type_list_str = ", ".join(node_type_list)
    edge_type_list_str = ", ".join(list(map(lambda x: "["+" ".join(x)+"]", edge_type_list))) # "[h r t]"
    central_node_type = node_type_list[0]
    num_node_types = len(node_type_list)
    other_node_type_and_graph_tokens = " ".join([f"\"{node_type}\": <graph>;" for node_type in node_type_list[1:]])
    format_instruction = deepcopy(format_template)
    format_instruction["graph_id"] = pyg_graph.graph_id
    format_instruction["conversations"][0]["value"] = format_instruction["conversations"][0]["value"].format(
        central_node_type=central_node_type,
        num_node_types=num_node_types,
        node_type_list=node_type_list_str,
        edge_type_list=edge_type_list_str,
        graph_raw_text=graph_raw_text,
        other_node_type_and_graph_tokens=other_node_type_and_graph_tokens,
        user_input=user_input
    )
    if few_shot_prompt is not None:
        format_instruction["conversations"][0]["value"] = few_shot_prompt + format_instruction["conversations"][0]["value"]

    format_instruction["conversations"][0]["value"] = CompactedStr(format_instruction["conversations"][0]["value"]).apply_formatting()

    return format_instruction