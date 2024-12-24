import sys
import os
from graph_agent import GraphActionAgent
from utils.storage import mem_store
from graph_tokenizer.graph_tokenizer import hetero_graph_tokenize
from graph_action_agent.agent import GraphActionAgent
from utils.args_parser import args
from task_planning_agent.agent import agent as task_planning_agent
from graph_generation_agent.agent import agent as graph_generation_agent
from utils.storage import mem_store
import torch
from graph_tokenizer.graph_tokenizer import hetero_graph_tokenize
from graph_tokenizer.build_input import build_input
from graph_action_agent.agent import GraphActionAgent
from colorama import Fore, Style

def read_user_instruction(source):
    if os.path.isfile(source):
        with open(source, 'r') as file:
            return file.read().strip()
    else:
        return source

def main():
    graph_action_agent = GraphActionAgent()

    while True:
        user_input = input("Please enter a user instruction or file path (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        user_instruction = read_user_instruction(user_input)

        res = task_planning_agent.invoke({"input": user_instruction})
        print(Fore.GREEN + "TASK PLANNING AGENT: " + str(res) + Style.RESET_ALL)

        res = graph_generation_agent.invoke(res)
        print(Fore.BLUE + "GRAPH GENERATION AGENT: " + str(res) + Style.RESET_ALL)

        grounded_graph = mem_store.mget(["pyg_graph"])[0]
        print(Fore.YELLOW + "GROUNDED GRAPH: " + str(grounded_graph) + Style.RESET_ALL)

        print(Fore.YELLOW + "GRAPH TOKENIZING... " + Style.RESET_ALL)
        grounded_graph_with_emb_gnn = hetero_graph_tokenize(grounded_graph)
        
        print(Fore.RED + "INVOKING GRAPH ACTION AGENT... " + Style.RESET_ALL)
        action_response = graph_action_agent.invoke(user_instruction, grounded_graph_with_emb_gnn, "generative")
        print(Fore.RED + "GRAPH ACTION AGENT: " + str(action_response) + Style.RESET_ALL)

if __name__ == "__main__":
    main()