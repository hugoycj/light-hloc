
FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends unzip wget libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip install --upgrade pip setuptools && \
    pip3 install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple && \
    rm -rf /root/.cache/pip/*

COPY . .

RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
     wget -O /root/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv-pth  https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth
RUN pip install typing-extensions --upgrade && pip install -e .

ENTRYPOINT ["hloc-process-data", "--data", "/data"]