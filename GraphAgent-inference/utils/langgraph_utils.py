from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from utils.storage import mem_store

def create_langgraph_agent(llm: ChatOpenAI, tools: list, system_prompt: str, output_parser=None):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    if len(tools) > 0:
        agent = create_openai_tools_agent(llm, tools, prompt)
    else:
        assert output_parser is not None, "You must provide an output parser if there are no tools."
        agent = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
            ) \
            | prompt | llm | output_parser
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def supervisor_node(state, supervisor, name):
    result = supervisor.invoke(state)
    # import pdb; pdb.set_trace()
    # state["next"] = result["next"]
    return {"next": result["next"]}

def agent_node(state, agent, name):
    # import pdb; pdb.set_trace()
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)], "worker_calling_steps": [name]}

def chain_node(state, chain, name):
    if name == "Graph Constructor":
        graph_input_keys = ["raw_text", "scaffold_nodes"]
        graph_input = {key: value for key, value in zip(graph_input_keys, mem_store.mget(graph_input_keys))}
        # import pdb; pdb.set_trace()
        result = chain.invoke({"input": graph_input})
    else:
        chain_input = state["messages"]
        result = chain.invoke({"input": chain_input})
        
    res_output = result.get("output", False)
    if not res_output:
        return {"messages": [HumanMessage(content=str(result), name=name)], "worker_calling_steps": [name]}
    else:
        return {"messages": [HumanMessage(content=res_output, name=name)], "worker_calling_steps": [name]}