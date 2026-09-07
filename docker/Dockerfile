FROM nvcr.io/nvidia/isaac-lab:2.3.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        unzip \
        nano \
    && rm -rf /var/lib/apt/lists/*

RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir -U "huggingface_hub[cli]"

RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
    defusedxml \
    onnxruntime \
    websockets \
    msgpack \
    msgpack-numpy \
    numpy

COPY . /workspace/green_challenge

WORKDIR /workspace/green_challenge

ENV PYTHONPATH=/workspace/green_challenge:$PYTHONPATH

RUN echo 'export PYTHONPATH=/workspace/green_challenge:$PYTHONPATH' >> /root/.bashrc

RUN echo 'alias python="/workspace/isaaclab/isaaclab.sh -p"' >> /root/.bashrc && \
echo 'alias pip="/workspace/isaaclab/isaaclab.sh -p -m pip"' >> /root/.bashrc && \
echo 'alias python3="/workspace/isaaclab/isaaclab.sh -p"' >> /root/.bashrc && \
echo 'alias pip3="/workspace/isaaclab/isaaclab.sh -p -m pip"' >> /root/.bashrc

RUN sed -i '\|name = "kit/community"|d' /isaac-sim/apps/*.kit && \
sed -i '\|name = "kit/community"|d' /workspace/isaaclab/apps/*.kit && \
sed -i '\|name = "kit/community"|d' /workspace/isaaclab/apps/isaacsim_4_5/*.kit

RUN /workspace/isaaclab/isaaclab.sh -p -m huggingface_hub.cli.hf download SberRoboticsCenter/GreenChallengeAssets \
  --repo-type dataset \
  --local-dir ./assets/

# Pre-warm Isaac Sim extension cache: the base image does not bundle the pinned
# isaacsim.asset.importer.urdf 2.4.31, so a first run would otherwise try to fetch
# it from cloudfront (breaks offline deployments).
RUN set -eux; \
    DEST="/root/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311"; \
    mkdir -p "$DEST"; \
    curl -fL --retry 3 -o /tmp/urdf-importer.zip \
        "http://d4i3qtqj3r0z5.cloudfront.net/isaacsim.asset.importer.urdf-2.4.31%2B107.3.3.lx64.r.cp311.zip"; \
    unzip -q /tmp/urdf-importer.zip -d "$DEST"; \
    rm /tmp/urdf-importer.zip; \
    test -d "$DEST/config"


ENTRYPOINT ["./run_benchmark.sh"]