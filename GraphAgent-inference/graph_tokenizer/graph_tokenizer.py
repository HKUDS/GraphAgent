import torch
import os


from graph_action_agent.graphllm.meta_hgt import MetaHGTConvCfg, MetaHGTConv
from graph_action_agent.graphllm.heteclip_models import CLIPTextCfg
import json
import os.path as osp
import glob
from lightning.pytorch import seed_everything
from sentence_transformers import SentenceTransformer

seed_everything(42)

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


sentence_transformer = SentenceTransformer(os.environ['SENTENCE_TRANSFORMER_MODEL_PATH'])

device = 'cuda:0'

def load_graph_tokenizer_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'graph_config.json')), 'graph_config.json missing'
    with open(osp.join(pretrain_model_path, 'graph_config.json'), 'r') as f:
        graph_config_dict = json.load(f)
    graph_cfg = MetaHGTConvCfg(**graph_config_dict)

    assert osp.exists(osp.join(pretrain_model_path, 'text_config.json')), 'text_config.json missing'
    with open(osp.join(pretrain_model_path, 'text_config.json'), 'r') as f:
        text_config_dict = json.load(f)
    text_cfg = CLIPTextCfg(**text_config_dict)
    
    assert model_name == MetaHGTConv
    model = model_name(in_channels = graph_cfg.in_channels,
        out_channels = graph_cfg.out_channels,
        heads = graph_cfg.heads,
        dynamic = graph_cfg.dynamic, 
        text_cfg = text_cfg,)

    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.ckpt'))
    state_dict = torch.load(pkl_files[0], map_location = 'cpu')['state_dict']
    print('Loading Graph Tokenizer ...')
    gnn_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.graph_encoder'):
            new_key = key.split('model.graph_encoder.')[1]
            gnn_state_dict[new_key] = value
    model.load_state_dict(gnn_state_dict, strict=False)

    return model

def build_meta_type_emb_dict(pyg_graph):
    node_type_emb_dict = {}
    for node_type in pyg_graph.node_types:
        node_type_emb_dict[node_type] = sentence_transformer.encode([node_type], convert_to_tensor=True)[0]
    edge_type_emb_dict = {}
    for edge_type in pyg_graph.edge_types:
        edge_type_emb_dict[edge_type] = sentence_transformer.encode([edge_type], convert_to_tensor=True)[0]
    return {
        "node_type_emb_dict": node_type_emb_dict,
        "edge_type_emb_dict": edge_type_emb_dict
    }

def encode_node_text(pyg_graph):
    # for each node, encode the node text into a 768-dim vector, and make it pyg_graph.x_dict
    x_dict = {}
    for node_type in pyg_graph.node_types:
        node_set = pyg_graph[node_type]
        """
        {'x': tensor([[0.]]), 'unified_idx': tensor([0]), 'description': ["Type: paper_contribution; Name: Mamba_model; Description: This paper introduces Mamba, a linear-time sequence model with selective state spaces. It modifies traditional state space models (SSMs) to be input-dependent and presents engineering techniques for enhanced performance. Experiments validate the method's effectiveness, offering various pre-trained model options."]}
        """
        x_dict_type_i = torch.zeros(len(node_set["description"]), 768)

        for i, node_text in enumerate(node_set["description"]):
            node_text_emb = sentence_transformer.encode([node_text], convert_to_tensor=True)[0]
            x_dict_type_i[i] = node_text_emb
        
        node_set["x"] = x_dict_type_i

        x_dict[node_type] = x_dict_type_i
    pyg_graph.x_dict = x_dict
    return pyg_graph

def hetero_graph_tokenize(pyg_graph): 
    pretrained_gnn_path = os.environ['GRAPH_TOKENIZE_MODEL_PATH']

    node_edge_type_emb_dict = build_meta_type_emb_dict(pyg_graph)
    pyg_graph = encode_node_text(pyg_graph)

    metahgt_model = load_graph_tokenizer_pretrained(MetaHGTConv, pretrained_gnn_path)
    metahgt_model = metahgt_model.to(device)
    
    node_feas_dict = node_edge_type_emb_dict["node_type_emb_dict"]
    for k, v in node_feas_dict.items():
        node_feas_dict[k] = v.to(device)
    
    edge_feas_dict = node_edge_type_emb_dict["edge_type_emb_dict"]
    for k, v in edge_feas_dict.items():
        edge_feas_dict[k] = v.to(device)
    
    pyg_graph = pyg_graph.to(device)
    for k in pyg_graph.edge_index_dict:
        pyg_graph[k].edge_index = pyg_graph[k].edge_index.to(torch.int64)

    node_type_feat_dict_i = {}
    for k in pyg_graph.x_dict:
        node_type_feat_dict_i[k] = node_feas_dict[k]
    
    edge_type_feat_dict_i = {}
    for k in pyg_graph.edge_index_dict:
        edge_type_feat_dict_i[k] = edge_feas_dict[k]

    with torch.no_grad():
        res = metahgt_model(x_dict = pyg_graph.x_dict,
            edge_index_dict = pyg_graph.edge_index_dict,  # Support both.
            node_type_feas_dict = node_type_feat_dict_i,
            edge_type_feas_dict = edge_type_feat_dict_i)
    
    pyg_graph.x_dict = res
    return pyg_graph