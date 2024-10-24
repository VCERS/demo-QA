#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from oxidation_potential import OxPredictor

def load_ox_potential_predictor():
  class ExampleOutput(BaseModel):
    oxidation_potential: float = Field(description = "oxidation potential prediction")  

  class OxPotentialPredictorInput(BaseModel):
    smiles: str = Field(description = "SMILES of a molecule")

  class OxPotentialPredictorConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: OxPredictor

  class OxPotentialPredictorTool(StructuredTool):
    name: str = "oxidation potential predictor"
    description: str = "predict oxidation potential of molecule in SMILES format"
    args_schema: Type[BaseModel] = OxPotentialPredictorInput
    config: OxPotentialPredictorConfig
    def _run(self, smiles: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ExampleOutput:
      ox = self.config.predictor.predict(smiles)
      return ExampleOutput(oxidation_potential = ox[0,0])

  predictor = OxPredictor()
  return OxPotentialPredictorTool(config = OxPotentialPredictorConfig(
    predictor = predictor
  ))

if __name__ == "__main__":
  tool = load_ox_potential_predictor()
  res = tool.invoke({'smiles': 'CCNCCNCCNCCNCCNCCNCC'})
  print(res)
