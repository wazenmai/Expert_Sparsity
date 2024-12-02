FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt update && apt install -y git wget vim
RUN pip install accelerate datasets fire tqdm
COPY ../lm-evaluation-harness lm-evaluation-harness
RUN pip install -e lm-evaluation-harness/
COPY ../transformers transformers
RUN pip install -e transformers/
COPY . .
