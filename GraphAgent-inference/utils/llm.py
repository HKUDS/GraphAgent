from langchain_openai import ChatOpenAI
from utils.args_parser import args

if args.planner == "gpt-3.5-turbo-0125":
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
elif args.planner == "deepseek":
    llm = ChatOpenAI(base_url="https://api.deepseek.com/v1", model="deepseek-chat", temperature=0.2)
