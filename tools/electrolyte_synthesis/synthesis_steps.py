#!/usr/bin/python3

from .prompts import exp_instruction_prompt
from ..reaction_path import PrecursorsRecommendation

class SynthesisSteps(object):
  def __init__(self, tokenizer, llm):
    self.recommend = PrecursorsRecommendation()
    template, parser = exp_instruction_prompt(tokenizer)
    self.chain = template | llm | parser
  def predict(self, query: str, n: int):
    predict = self.recommend.call(target_formula = [query], top_n = n)[0]
    precursors_predicts = predict['precursors_predicts']
    results = list()
    for idx, precursors in enumerate(precursor_sets):
      steps = chain.invoke({'precursors': ','.join(precursors), 'target': query})
      results.append(steps)
    return results


