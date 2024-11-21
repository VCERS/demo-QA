#!/usr/bin/python3

import requests
from langchain.llms.base import LLM
from langchain_community.llms import HuggingFaceEndpoint
'''
class Qwen2(LLM):
  url: str = None
  headers: dict = None
  def __init__(self, host):
    super(Qwen2, self).__init__()
    self.url = host
    self.headers = {'Authorization': "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"}
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    data = {"inputs": prompt, "parameters": {"temperature": 0.6, "top_p": 0.9}}
    for i in range(10):
      response = requests.post(self.url, headers = self.headers, json = data)
      if response.status_code == 200:
        break
    else:
      raise Exception(f'请求失败{response.status_code}')
    return response.json()['generated_text']
  @property
  def _llm_type(self):
    return "tgi"
'''
def Qwen2(*args,**kwargs):
  return HuggingFaceEndpoint(
    endpoint_url = "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation",
    do_sample = False,
    top_p = 0.9,
    temperature = 0.6,
    trust_remote_code = True,
    use_cache = True
  )

