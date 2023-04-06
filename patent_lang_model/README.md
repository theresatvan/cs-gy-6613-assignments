---
title: Patent Language Model
emoji: 🐨
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

## Hugging 🤗 Face Space
<https://huggingface.co/spaces/theresatvan/patent-language-model>

## Containerization setup
We use Docker for containerization. Since we are building a Streamlit application, we can expect to be developing solely in Python. The Docker base image we are building our container from is `python:3.10-slim`.

[Reference for deploying Streamlit application in Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

We then use VS Code Dev Containers to allow us to create a Docker image from the Dockerfile and develop inside of a Docker container. 

[Reference for setting up VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container#_dockerfile)

Our container configuration will be set up as such:
```
└── patent_lang_model
    └── .devcontainer
        ├── devcontainer.json
        └── Dockerfile
```
