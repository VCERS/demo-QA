#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .reaction_path import PrecursorsRecommendation

def load_precursor_predictor():
  class ExampleOutput(BaseModel):
    target_formula: str = Field(description = "the input chemical expressin")
    precursors_predicts: list = Field(description = "precursor combinations each of which is saved in a tuple in the list")

  class PrecursorPredictorInput(BaseModel):
    query: str = Field(description = "chemical expression of a compound")
    n: int = Field(description = "how many precursor combinations are returned. if not specified just use value 1.")

  class PrecursorPredictorConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    recommend: PrecursorsRecommendation

  class PrecursorPredictorTool(StructuredTool):
    name: str = "precursor predictor"
    description: str = "predict multiple possible precursor combinations of a compound"
    args_schema: Type[BaseModel] = PrecursorPredictorInput
    config: PrecursorPredictorConfig
    def _run(self, query:str, n: int = 1, run_manager: Optional[CallbackManagerForToolRun] = None) -> ExampleOutput:
      target_formula = [query]
      all_predicts = self.config.recommend.call(target_formula = target_formula, top_n = n)
      return ExampleOutput(**all_predicts[0])

  recommend = PrecursorsRecommendation()
  return PrecursorPredictorTool(config = PrecursorPredictorConfig(
    recommend = recommend
  ))

if __name__ == "__main__":
  tool = load_precursor_predictor()
  res = tool.invoke({'query': 'BaYSi2O5N','n': 10})
  print(res)
