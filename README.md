# 🏌️‍♂️ Golf Action Analysis API with STGCN (Based on mmaction2)

This repository details the modifications and deployment steps for a custom action recognition project, forked from OpenMMLab's **mmaction2**. We specifically leveraged the **STGCN (Spatial Temporal Graph Convolutional Networks)** model and fine-tuned it for precise golf swing analysis. The final, optimized model is packaged and served as a scalable RESTful API using **Docker** for standardized deployment.

## 🚀 Key Project Customizations

- **Framework Adaptation:** Forked and extensively modified the `mmaction2` codebase to integrate custom data loaders, configs, and training scripts specific to golf action sequences.
- **Model Fine-tuning:** Applied transfer learning to the STGCN architecture using a dedicated golf action dataset. Transfer learned inference is a binary classification of true or false.
- **Containerization (Docker):** Created a comprehensive deployment environment using a custom `Dockerfile`, ensuring all dependencies (PyTorch, mmaction2, etc.) and the fine-tuned model checkpoint are encapsulated.
- **API Server Implementation:** Developed a robust RESTful API layer (e.g., using FastAPI) within the Docker container to handle keypoint data submissions and return classification or localization results.

## 🛠️ Tech Stack

| Category | Technology/Tool | Purpose |
| :--- | :--- | :--- |
| **Base Framework** | `mmaction2` (Modified) | Action Recognition Backbone |
| **Model** | **STGCN** | Primary Algorithm for Golf Action Analysis |
| **Deployment** | **Docker** | Environment Standardization and Isolation |
| **Language** | **Python** | Development and API Logic |
| **API Backend** | (FastAPI) | Exposing Model Inference via HTTP |

## Docker API

https://github.com/ChanGyu-Cho/mmaction2-stgcn-golf-api

Check this docker repo, to Make api server

For docker setting, refer to the contents of the docker folder. Set up various heavy packages and basics through dockerfile.base, and configure mmaction configuration, models, and servers through dockerfile.
