#!/usr/bin/python3

from .prompts import search_prompt

class KGSearch(object):
  def __init__(self, tokenizer, llm):
    template, parser = search_prompt(tokenizer)
    self.chain = template | llm | parser
  def extract(self, query: str):
    results = self.chain.invoke({'query': query})
    return results


