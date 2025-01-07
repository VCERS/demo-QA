#!/usr/bin/python3
import os
from absl import flags, app
import gradio as gr
from agent import Agent
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
os.environ['HF_HOME'] = config.get('General', 'huggingface')
os.environ["CUDA_VISIBLE_DEVICES"] = config.get('General', 'device')

huggingface_token = config.get('General', 'huggingface_token')
LANGCHAIN_key = config.get('General', 'LANGCHAIN_key')

service_host = "0.0.0.0"
service_port = 19214

import os
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_key


FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama3', 'qwen2', 'NV_llama'}, help = 'model to use')

def create_interface():
  # Agent automatically loads the model and a sutiable tool
  agent = Agent(model = FLAGS.model)

  def chatbot_response(user_input, history):
    response = agent.query(user_input)
    history.append((user_input, response['output']))
    return "", history, history
  def clear_chatbot_memory():
      agent.clear()
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>Electrolyte Agent</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        user_input = gr.Textbox(label = 'What is your question？')
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot, state], value = "Clear Chat")
          submit_btn = gr.Button("Send")
      user_input.submit(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [user_input, state, chatbot])
      submit_btn.click(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [user_input, state, chatbot])
      clear_btn.click(clear_chatbot_memory)
      clear_btn.click(
          lambda _: gr.State([]),
          [state],
          [state],
      )
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = service_host, server_port = service_port, share=True)

if __name__ == "__main__":
  add_options()
  app.run(main)
