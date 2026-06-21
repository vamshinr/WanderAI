# WanderAI demo — HuggingFace Docker Space.
# Serves the browser UI (serve.py) for the 2D scenes. The trained models live on
# Fireworks; this container calls them server-side using the FIREWORKS_API_KEY that
# you set as a Space *secret* (never baked into the image). 3D (MuJoCo) scenes need
# GL and are not enabled in this slim image — run those locally.
FROM python:3.12-slim

WORKDIR /app
# Pure-python + numpy environment; certifi gives the Fireworks TLS calls a CA bundle.
RUN pip install --no-cache-dir numpy certifi

# App code only — no .env (secret), no 3D scene assets, no venvs (see .dockerignore).
COPY wanderai ./wanderai
COPY serve.py ./
COPY ui ./ui

# HF Spaces route to 0.0.0.0:7860. FIREWORKS_API_KEY is injected from Space secrets.
ENV HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "serve.py"]
