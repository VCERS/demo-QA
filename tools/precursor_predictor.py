#!/usr/bin/python3

from pydantic import BaseModel, Field
from .reaction_path import PrecursorsRecommendation

def load_precursor_predictor():
  class PrecursorPredictorInput(BaseModel):
    query: str = Field(description = "chemical expression of a compund")
    n: int = Field(description = "how many precursor combinations are returned. if not specified just use value 1.")

  class PrecursorPredictorConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    recommend: PrecursorsRecommendation

  class PrecursorPredictorTool(BaseTool):
    name: str = "precursor predictor"
    description: str = "predict multiple possible precursor combinations of a compound"
    args_schema: Type[BaseModel] = PrecursorPredictorInput
    return_direct: bool = True
    config: PrecursorPredictorConfig
    def _run(self, query:str, n: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      target_formula = [query]
      all_predicts = recommend.call(target_formula = target_formula, top_n = n)
      return str(all_predicts[0]['precursors_predicts'])

  recommend = PrecursorsRecommendation()
  return PrecursorPredictorTool(config = PrecursorPredictorConfig(
    recommend = recommend
  ))

