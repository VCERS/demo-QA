#!/usr/bin/python3

huggingface_token = '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'
LANGCHAIN_key = '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'

service_host = "0.0.0.0"
service_port = 8081


import os
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_key