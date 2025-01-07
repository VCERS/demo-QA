#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List
from langchain.tools import StructuredTool, tool
from langchain.callbacks.manager import CallbackManagerForToolRun
from .kg_search import KGSearch, graph_query
from neo4j import GraphDatabase

host = "bolt://103.6.49.76:7687"
password = "19841124"
user = "neo4j"
db = "mergetest"
driver = GraphDatabase.driver(host, auth=(user, password))

def load_kg_search(tokenizer, llm):
    
  class ELT(BaseModel):
    elt: str = Field(description = "Electrolyte which is usually in a form of chemical expression.")

  class KGSearchInput(BaseModel):
    query: str = Field(description = "A questions about a compound in the form of chemical expression.")
      
  class KGSearchExtractorConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    extractor: KGSearch
      
  class KGSearchTool(StructuredTool):
    name: str = "Electrolyte synthesis path extractor"
    description: str = "Extract chemical expression of electrolyte from query if user asked the synthesis path of any electrolyte."
    args_schema: Type[BaseModel] = KGSearchInput
    config: KGSearchExtractorConfig
    def _run(self, query: str) -> ELT:
      elts = self.config.extractor.extract(query)
      query = elts['elt']
      text = graph_query(query, driver)
      return text

  extractor = KGSearch(tokenizer, llm)
  return KGSearchTool(config = KGSearchExtractorConfig(extractor = extractor))

if __name__ == "__main__":
  tools = load_kg_search(tokenizer, llm)
  res = tool.invoke({'query': 'What is the synthesis path of Li6PS5Cl?'})
  print(res)
