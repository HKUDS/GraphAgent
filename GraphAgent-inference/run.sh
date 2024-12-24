export OPENAI_API_KEY="" # set your Planner API key here
export SENTENCE_TRANSFORMER_MODEL_PATH="sentence-transformers/all-mpnet-base-v2"
export GRAPH_ACTION_MODEL_PATH="GraphAgent/GraphAgent-7B"
export GRAPH_TOKENIZE_MODEL_PATH="GraphAgent/GraphTokenizer"

PLANNER="deepseek"

python serve_graph_agent.py --planner $PLANNER