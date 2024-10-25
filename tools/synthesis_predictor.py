#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .electrolyte_synthesis import SynthesisSteps

def load_synthesis_steps_predictor(tokenizer, llm):
  class SynthesisProcedure(BaseModel):
    steps: List[str] = Field(description = "a list of steps of one synthesis procedure")

  class ExampleOutput(BaseModel):
    synthesis_procedures: List[SynthesisProcedure] = Field(description = "a list of synthesis procedures")

  class SynthesisStepsPredictorInput(BaseModel):
    query: str = Field(description = "chemical expression of a compound")
    n: int = Field(description = "how many procedures are returned. if not specified just use value 1.")

  class SynthesisStepsPredictorConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: SynthesisSteps

  class SynthesisStepsPredictorTool(StructuredTool):
    name: str = "electrolyte synthesis procedure predictor"
    description: str = "predict multiple possible procedures of an electrolyte synthesis"
    args_schema: Type[BaseModel] = SynthesisStepsPredictorInput
    config: SynthesisStepsPredictorConfig
    def _run(self, query: str, n: int = 1, run_manager: Optional[CallbackManagerForToolRun] = None) -> ExampleOutput:
      procedures = self.config.predictor.predict(query, n)
      return ExampleOutput(synthesis_procedures = procedures)

  predictor = SynthesisSteps(tokenizer, llm)
  return SynthesisStepsPredictorTool(config = SynthesisStepsPredictorConfig(
    predictor = predictor
  ))

if __name__ == "__main__":
  tools = load_synthesis_steps_predictor(tokenizer, llm)
  res = tool.invoke({'query': 'Li6PS5Cl', 'n': 2})
  print(res)
