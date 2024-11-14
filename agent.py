#!/usr/bin/python3

from os import environ
from transformers import AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from models import Qwen2
from prompts import agent_template
from tools import load_precursor_predictor, load_ox_potential_predictor, load_synthesis_steps_predictor
import config

class Agent(object):
  def __init__(self, host = 'http://localhost:8080/generate', tools = ['google-serper', 'llm-math', 'wikipedia', 'arxiv']):
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    llm = Qwen2(host)
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + \
            [load_precursor_predictor(),
             load_ox_potential_predictor(),
             load_synthesis_steps_predictor(tokenizer, llm)]
    prompt = agent_template(tokenizer)
    llm = llm.bind(stop = ["<|eot_id|>"])
    chain = {
      "input": lambda x: x["input"],
      "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
      "tools": render_text_description(tools),
      "tool_names": ", ".join([t.name for t in tools])
    } | prompt | llm | ReActJsonSingleInputOutputParser()
    memory = ConversationBufferMemory(memory_key="chat_history")
    self.agent_chain = AgentExecutor(agent = chain, tools = tools, memory = memory, verbose = True, handle_parsing_errors=True)
  def query(self, question):
    return self.agent_chain.invoke({"input": question})

if __name__ == "__main__":
  agent = Agent(host = 'http://localhost:8080/generate')
  print(agent.query("Give 5 precursor combinations of SrZnSO"))
