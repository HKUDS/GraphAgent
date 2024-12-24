import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--planner", type=str, default="deepseek")
parser.add_argument("--executor", type=str, default="graphagent")

args = parser.parse_args()