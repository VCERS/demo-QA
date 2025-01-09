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


service_host = config.get('General', 'service_host')
service_port = int(config.get('General', 'service_port'))

import os
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_key


FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama3', 'qwen2', 'NV_llama'}, help = 'model to use')
  flags.DEFINE_string('is_Neo', default='False', help='if we want to use neo4j based method.')
  flags.DEFINE_string('service_port', default=config.get('General', 'service_port'), help='port.')

def create_interface():
  # Agent automatically loads the model and a sutiable tool
  if FLAGS.is_Neo == 'True':
      agent = Agent(model = FLAGS.model, is_neo=True)
  else:
      agent = Agent(model = FLAGS.model, is_neo=False)

  def chatbot_response(user_input, history):
    response = agent.query(user_input)
    history.append((user_input, response['output']))
    return "", history, history
  def clear_chatbot_response(user_input, history):
    return "", []
  def clear_chatbot_memory():
      agent.clear()
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        if FLAGS.is_Neo == 'True':
            gr.Markdown("<h1><center>Electrolyte Agent Neo4j Based</center></h1>")
        else:
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
      clear_btn.click(clear_chatbot_response,
                     inputs = [user_input, state],
                     outputs = [user_input, state])
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = service_host, server_port = int(FLAGS.service_port), share=True)

if __name__ == "__main__":
  add_options()
  app.run(main)
