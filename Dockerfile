# Pinned so every environment runs the same vLLM. Cloud Build passes a mirror.gcr.io base
# to avoid Docker Hub rate limits.
ARG BASE=vllm/vllm-openai:v0.30.0
FROM ${BASE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Palimpzest gets its own environment: its pinned dependencies (numpy, litellm) must not change
# vLLM's. It reuses the image's torch through the system site-packages.
COPY pz-constraints.txt .
RUN python3 -m venv --system-site-packages /opt/pz \
    && /opt/pz/bin/pip install --no-cache-dir -c pz-constraints.txt palimpzest==1.5.3

COPY harness/ harness/
COPY configs/ configs/

# Fails the build if the Palimpzest environment cannot load the query module.
RUN /opt/pz/bin/python -c "import harness.pz_query"

# Model weights are mounted at /models, not baked into the image.
ENV HF_HOME=/models
# Enables /sleep, /wake_up and /reset_prefix_cache.
ENV VLLM_SERVER_DEV_MODE=1
ENV PYTHONUNBUFFERED=1

# The runner starts the vLLM servers itself.
ENTRYPOINT ["python3", "-m", "harness.runner"]
CMD ["--config", "configs/smoke.yaml"]
