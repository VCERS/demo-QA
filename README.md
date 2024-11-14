# Introduction

this project demo what we can do about electrolyte synthesis with LLM

## Tools

the agent can call the following tools automatically

| tool | description |
|------|-------------|
| oxidation potential predictor | predict oxidation potential of molecule in SMILES format |
| precursor predictor | predict multiple possible precursor combinations of a compound |
| electrolyte synthesis procedure predictor | predict multiple possible procedures of an electrolyte synthesis |
| google-serper | google search tool |
| llm-math | math problem solving tool |
| wikipedia | wikipedia search tool |
| arxiv | arxiv search tool |

# Usage

## install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## deploy text-generate-inference

```shell
model=Qwen/Qwen2.5-7B-Instruct
docker pull ghcr.io/huggingface/text-generation-inference
docker run --gpus all --shm-size 1g -p 8080:80 -v <home>/.cache/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model
```

## run server

```shell
python3 main.py
```
