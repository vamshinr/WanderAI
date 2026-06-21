# Deploy the WanderAI demo to a HuggingFace Space

The repo is Space-ready:
- `Dockerfile` — serves the browser UI (`serve.py`) on port 7860.
- `.dockerignore` — keeps the image small and **secret-free** (no `.env`, no venvs, no 3D assets).
- `README.md` frontmatter — HF Space metadata (`sdk: docker`, `app_port: 7860`).

The trained models stay on **Fireworks**; the Space calls them server-side with a key
you provide as a **secret** (never committed, never sent to the browser).

## 1. Create the Space + add the secret
1. https://huggingface.co/new-space → **SDK: Docker**, name e.g. `wanderai`.
2. Space → **Settings → Variables and secrets → New secret**:
   - name: `FIREWORKS_API_KEY`
   - value: your Fireworks key

## 2. Push (secret-safe — git respects `.gitignore`, so `.env` stays local)
```bash
git remote add space https://huggingface.co/spaces/<your-username>/wanderai
git push space HEAD:main
```
HF builds the Dockerfile and serves it at `https://<your-username>-wanderai.hf.space`
— that's your demo link.

## Notes for judges / demo
- **2D scenes work fully** on the hosted link (pick a scene → **Trained**; default model
  is **st1**, the proven 2D model; use the dropdown to compare models).
- **3D scenes** need MuJoCo's GL rendering, which isn't in this slim image — show 3D by
  running locally: `python3 serve.py` → **Load 3D Test scene**.
- Fireworks deployments are scale-to-zero, so the first **Trained** click cold-starts the
  model (a few seconds), then it's fast.
