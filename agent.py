#!/usr/bin/python3

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain import hub
from langchain.tools.render import render_text_description
from langchain_core.prompts.prompt import PromptTemplate
from models import Llama3, Qwen2, Nvidia_llama
from tools import load_precursor_predictor, load_ox_potential_predictor, load_synthesis_steps_predictor, load_kg_search, load_kg_search_neo



def agent_template(tokenizer, tools):
  prompt = hub.pull('yzprompt/personal_test')
  #print(prompt)
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
  def __init__(self, model = 'llama3', tools = ['google-serper', 'llm-math', 'wikipedia', 'arxiv'], is_neo = False):
    llms_types = {
      'llama3': Llama3,
      'qwen2': Qwen2,
      'NV_llama': Nvidia_llama,
    }
    tokenizer, llm = llms_types[model](True)
    self.tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + [load_kg_search(tokenizer, llm)]
    if is_neo:
        self.tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + [load_kg_search_neo(tokenizer, llm)]
            #[load_precursor_predictor(),
            # load_ox_potential_predictor(),
            # load_synthesis_steps_predictor(tokenizer, llm)]
    prompt = agent_template(tokenizer, self.tools)
    llm = llm.bind(stop = ["<|eot_id|>"])
    self.chain = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])} | prompt | llm | ReActJsonSingleInputOutputParser()
    # memory = ConversationBufferMemory(memory_key="chat_history")
    self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    
    self.agent_chain = AgentExecutor(agent = self.chain, tools = self.tools, memory = self.memory, verbose = True, handle_parsing_errors=False, max_iterations=4)
  def query(self, question):
    return self.agent_chain.invoke({"input": question})
  def clear(self):
    self.memory.clear()
    self.agent_chain = AgentExecutor(agent = self.chain, tools = self.tools, memory = self.memory, verbose = True, handle_parsing_errors=False, max_iterations=4)

if __name__ == "__main__":
  agent = Agent(model = "NV_llama")
  print(agent.query("What is the synthesis path of Li7P3S11?"))
