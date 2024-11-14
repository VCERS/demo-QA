#!/usr/bin/python3

from typing import Any, Union
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue

class HFChatPromptValue(ChatPromptValue):
  tokenizer: Any = None
  def to_string(self) -> str:
    hf_messages = []
    for m in self.messages:
      if isinstance(m, HumanMessage):
        role = 'user'
      elif isinstance(m, AIMessage):
        role = 'assistant'
      elif isinstance(m, SystemMessage):
        role = 'system'
      else:
        raise Exception(f'Got unsupported message type: {m}')
      hf_messages.append({'role': role, 'content': m.content})
    return self.tokenizer.apply_chat_template(hf_messages, tokenize = False, add_generation_prompt = True)

class HFChatPromptTemplate(ChatPromptTemplate):
  tokenizer: Any = None
  def format_prompt(self, **kwargs: Any) -> PromptValue:
    messages = self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)
  async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
    messages = await self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)

def exp_instruction_prompt(tokenizer):
  class UnstructuredSteps(BaseModel):
    steps: list = Field(
      description = (
        "experiment step instructions given in human readable strings as elements of this list"
      )
    )
  parser = JsonOutputParser(pydantic_object = UnstructuredSteps)
  examples = [
    {
      "precursors" : ["Li2O","Li2S","Li3PO4","LiCl"],
      "target": "Li6PS5Cl",
      "steps": [
        "准备zirconia grinding jar类型45ml容器c1。",
        "向容器c1添加0.464g前体物质Li2O。",
        "向容器c1添加0.591g前体物质Li2S。",
        "向容器c1添加0.360g前体物质Li3PO4。",
        "向容器c1添加0.085g前体物质LiCl。",
        "采用planetary ball mill对容器c1内的样本进行研磨，研磨转速为370rpm，研磨时间为90000秒。",
        "准备Tammann tube类型，PT2型号的容器c2。",
        "在argon环境的Glove Box中将容器c1中的物质转移1.5g到容器c2中。",
        "在argon环境的Glove Box中对容器c2采用quartz wool封闭。",
        "在argon环境的Glove Box中对容器c2采用SUS sealed container封闭。",
        "在argon环境的Glove Box中对容器c2进行加热，从室温到目标温度为520C，加热需要在10800秒时间以2.5/min速率加热到目标温度，在目标温度需要维持28800秒。",
        "在argon环境的Glove Box中对容器c2进行冷却，到室温。"
      ]
    },
    {
      "precursors": ['Li3N', 'LiCl'],
      "target": 'Li6N0.3Cl0.7',
      "steps": [
        "准备quartz tube类型50ml容器c1。",
        "在nitrogen环境的Glove Box中对容器c1添加40mol的Li3N。",
        "在nitrogen环境的Glove Box中对容器c1添加60mol的LiCl。",
        "准备molybdenum crucible类型1000ml容器c2。",
        "在nitrogen环境的Glove Box中从容器c1中取出样本用100MPa压力压成片，片的厚度为10mm，直径为5mm。将这些片状样本转移到容器c2。",
        "在nitrogen环境的Glove Box中对容器c2进行加热，从室温到目标温度为435C，加热需要在300秒时间以100/min速率加热到目标温度，在目标温度需要维持72000秒。"
      ]
    },
    {
      "precursors": ['LiCl', 'YCl', 'InCl'],
      "target": "Li6N0.3Cl0.7",
      "steps": [
        "准备zirconia grinding jar类型100ml容器c1。",
        "在vacuum环境的Glove Box中对容器c1添加3.81g的LiCl。",
        "在vaccuum环境的Glove Box中对容器c1添加4.68g的YCl。",
        "在vaccuum环境的Glove Box中对容器c1添加1.32g的InCl。",
        "在vaccuum环境的Glove Box中对容器c1采用rubber sealing gasket封闭。",
        "采用planetary ball mill对容器c1内的样本进行研磨，研磨转速为550rpm，研磨时间为36000秒。",
        "准备quartz tube类型600ml容器c2。",
        "在vaccuum环境的Glove Box中从容器c1中提取10份，每份重量为2g的样本，每份压成薄片并转移到容器c2。",
        "在vaccuum环境的Glove Box中对容器c2采用tube sealing machine封闭。",
        "对容器c2进行煅烧，煅烧在温度260C持续36000秒。升温速度为2/min。冷却速度为2/min。"
      ]
    },
    {
      "precursors": ["Li2S", "P2S5", "LiBr", "LiI", "MnS"],
      "target": "Li6N0.3Cl0.7",
      "steps": [
        "准备zirconia grinding jar类型45ml容器c1。",
        "向容器c1添加0.5450g前体物质Li2S。",
        "向容器c1添加0.8857g前体物质P2S5。",
        "向容器c1添加0.2768g前体物质LiBr。",
        "向容器c1添加0.2844g前体物质LiI。",
        "向容器c1添加0.0081g前体物质MnS。",
        "向容器c1添加4.0g前体物质toluene。",
        "采用planetary ball mill对容器c1内的样本进行研磨，研磨转速为500rpm，研磨时间为144000秒。",
        "在argon环境的Glove Box中对容器c1添加8.6g的toluene。",
        "准备bucket for plate heater类型45ml容器c2。",
        "在argon环境的Glove Box中从容器c1转移样本到容器c2。",
        "在argon环境的Glove Box中对容器c2进行沉淀，然后去除supernatant solvent。",
        "在argon环境的Glove Box中对容器c2进行加热，从室温到目标温度为80C，在目标温度需要维持10800秒。"
      ]
    },
    {
      "precursors": ['Li2S', 'P2S5', 'Br2', 'I2'],
      "target": "Li6N0.3Cl0.7",
      "steps": [
        "准备zirconia grinding jar类型45ml容器c1。",
        "向容器c1添加0.661g前体物质Li2S。",
        "向容器c1添加0.914g前体物质P2S5。",
        "向容器c1添加0.164g前体物质Br2。",
        "向容器c1添加0.261g前体物质I2。",
        "向容器c1添加4g前体物质chlorobenzene。",
        "采用planetary ball mill对容器c1内的样本进行研磨，研磨转速为500rpm，研磨时间为144000秒。",
        "向容器c1添加20ml前体物质chlorobenzene。",
        "准备schlenk bottle类型50ml容器c2。",
        "从容器c1转移样本到容器c2。",
        "对容器c2通过沉淀方法，去除supernatant solvent。",
        "对容器c2加热到100C，并通过vacuum pump进行干燥。",
        "在argon环境的Glove Box中对容器c2进行加热，从室温到目标温度为188C，在目标温度需要维持10800秒。"
      ]
    }
  ]
  instructions = parser.get_format_instructions()
  instructions = instructions.replace('{','{{')
  instructions = instructions.replace('}','}}')
  examples = str(examples).replace('{','{{')
  examples = examples.replace('}','}}')
  system_prompts = """You are an electrolyte design expert. You can design the experimental steps for electrolyte synthesis based on the given precursors of an electrolyte and your own chemical knowledge.


Below are a number of examples of given precursors, target electrolyte and their generated experimental steps.
%s""" % examples
  human_prompts = """For the following given precursors and target electrolyte, design an experiment in detail steps.
%s

precursors: {precursors}
target: {target}""" % instructions
  template = HFChatPromptTemplate([
    ('system', system_prompts),
    ('user', human_prompts)
  ], tokenizer = tokenizer)
  return template, parser
