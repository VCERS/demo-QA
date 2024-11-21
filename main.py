#!/usr/bin/python3

import time
from absl import flags, app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import gradio as gr
from agent import Agent
import config

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = 'http://localhost:8080/generate', help = 'url to TGI')

def create_interface():
  agent = Agent(host = FLAGS.host)
  def chatbot_response(user_input, history):
    chat_history = list()
    for human, ai in history:
      chat_history.append(HumanMessage(content = human))
      chat_history.append(AIMessage(content = ai))
    chat_history = chat_history[-config.max_history_len * 2:]
    response = agent.query(user_input, chat_history)
    history.append((user_input, response['output']))
    return "", history, history
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>Electrolyte Agent</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        user_input = gr.Textbox(label = '需要问什么？')
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot, state], value = "清空问题")
      submit_btn.click(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [user_input, state, chatbot])
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = config.service_host,
              server_port = config.service_port)

if __name__ == "__main__":
  add_options()
  app.run(main)
