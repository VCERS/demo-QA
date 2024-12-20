#!/usr/bin/python3

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain import hub
from langchain.tools.render import render_text_description
from langchain_core.prompts.prompt import PromptTemplate
from models import Llama3, Qwen2
from tools import load_precursor_predictor, load_ox_potential_predictor, load_synthesis_steps_predictor


def agent_template(tokenizer, tools):
  prompt = hub.pull('hwchase17/react-json')
  system_template = prompt[0].prompt.template
  system_template = system_template.replace('{tools}', render_text_description(tools))
  system_template = system_template.replace('{tool_names}', ", ".join([t.name for t in tools]))
  user_template = prompt[1].prompt.template
  messages = [
    {'role': 'system', 'content': system_template},
    {'role': 'user', 'content': user_template}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['agent_scratchpad', 'input'])
  return template


class Agent(object):
  def __init__(self, model = 'llama3', tools = ['google-serper', 'llm-math', 'wikipedia', 'arxiv']):
    llms_types = {
      'llama3': Llama3,
      'qwen2': Qwen2
    }
    tokenizer, llm = llms_types[model](True)
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + \
            [load_precursor_predictor(),
             load_ox_potential_predictor(),
             load_synthesis_steps_predictor(tokenizer, llm)]
    prompt = agent_template(tokenizer, tools)
    llm = llm.bind(stop = ["<|eot_id|>"])
    chain = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])} | prompt | llm | ReActJsonSingleInputOutputParser()
    # memory = ConversationBufferMemory(memory_key="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    
    self.agent_chain = AgentExecutor(agent = chain, tools = tools, memory = memory, verbose = True, handle_parsing_errors=True)
  def query(self, question):
    return self.agent_chain.invoke({"input": question})

if __name__ == "__main__":
  agent = Agent(model = "llama3")
  print(agent.query("Give 5 precursor combinations of SrZnSO"))
