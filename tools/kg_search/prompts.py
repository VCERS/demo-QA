#!/usr/bin/python3

from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate

def search_prompt(tokenizer):
  class SearchItem(BaseModel):
    elt: str = Field(description = "Electrolyte appeared in the sentence.")
  parser = JsonOutputParser(pydantic_object = SearchItem)
  examples = [
    {
      "query" : "What is the synthesis path for Li6PS5CL?",
      "output": "Li6PS5Cl"
    },
    {
      "query" : "What is the synthesis path for Li3PS4?",
      "output": "Li3PS4"
    },
    {
      "query" : "What is the synthesis path for Li4SnS4?",
      "output": "Li4SnS4"
    },
    {
      "query" : "What is the synthesis path for LSPSCl?",
      "output": "LSPSCl"
    },
    {
      "query" : "What is the synthesis path for Li7P3S11?",
      "output": "Li7P3S11"
    }
  ]
  instructions = parser.get_format_instructions()
  instructions = instructions.replace('{','{{')
  instructions = instructions.replace('}','}}')
  examples = str(examples).replace('{','{{')
  examples = examples.replace('}','}}')
  system_prompts = """You are an electrolyte extract expert. You can extract electrolyte from user's questions.


Below are a number of examples how you should respond to user's questions.
%s""" % examples
  human_prompts = """For the following question if it ask about the synthesis path of any electrolyte, extract all electrolyte from the question.
%s

query: {query}
""" % instructions
  messages = [
    {'role': 'system', 'content': system_prompts},
    {'role': 'user', 'content': human_prompts}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['query'])
  return template, parser
