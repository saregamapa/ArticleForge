# ArticleForge – Deployment Guide

Your app is a **single FastAPI process** that serves the API and all static HTML. You only need to run the backend and set `OPENAI_API_KEY`.

---

## Option 1: Railway (easiest, good free tier)

1. Go to [railway.app](https://railway.app), sign in with GitHub.
2. **New Project** → **Deploy from GitHub repo** → select `saregamapa/ArticleForge`.
3. Add environment variable: `OPENAI_API_KEY` = your OpenAI key.
4. Railway will detect Python and run the app. If it doesn’t auto-detect, set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn backend:app --host 0.0.0.0 --port $PORT`
5. Deploy. Railway gives you a URL like `https://articleforge-production.up.railway.app`.

**Pros:** Very simple, GitHub auto-deploys, free tier.  
**Cons:** Sleep on free tier; first request after idle can be slow.

---

## Option 2: Render (simple, free tier)

1. Go to [render.com](https://render.com), sign in with GitHub.
2. **New** → **Web Service** → connect `saregamapa/ArticleForge`.
3. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn backend:app --host 0.0.0.0 --port $PORT`
4. **Environment** → add `OPENAI_API_KEY`.
5. Deploy. You get a URL like `https://articleforge.onrender.com`.

**Pros:** Straightforward, free tier, GitHub deploys.  
**Cons:** Free tier spins down after inactivity; cold starts.

---

## Option 3: Docker (run anywhere: VPS, Cloud Run, etc.)

A **Dockerfile** is included. Build and run locally:

```bash
docker build -t articleforge .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here articleforge
```

**Deploy the image:**

- **Google Cloud Run:** Push image to Artifact Registry or Docker Hub, then create a Cloud Run service (set `OPENAI_API_KEY` as env var). Scales to zero, pay per request.
- **AWS (ECS / App Runner):** Push to ECR, create service; set env and port 8000.
- **Your own VPS (DigitalOcean, Linode, etc.):** Install Docker on the server, run the same `docker run` (or use docker-compose) and put Nginx in front if you want HTTPS/domain.

**Pros:** Same setup everywhere; easy to move between providers.  
**Cons:** You manage the server or the cloud project.

---

## Option 4: VPS without Docker (Ubuntu/Debian)

On a fresh VPS (e.g. DigitalOcean droplet):

```bash
# Install Python 3.11, pip, and create app dir
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip
sudo useradd -m -s /bin/bash app
sudo su - app
git clone https://github.com/saregamapa/ArticleForge.git
cd ArticleForge
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
# Run with a process manager (see below)
```

Run the app with **systemd** (so it restarts on crash and on reboot):

```ini
# /etc/systemd/system/articleforge.service
[Unit]
Description=ArticleForge
After=network.target

[Service]
User=app
WorkingDirectory=/home/app/ArticleForge
EnvironmentFile=/home/app/ArticleForge/.env
ExecStart=/home/app/ArticleForge/.venv/bin/uvicorn backend:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then: `sudo systemctl daemon-reload && sudo systemctl enable --now articleforge`. Put **Nginx** (or Caddy) in front for HTTPS and your domain.

**Pros:** Full control, no cold starts, predictable cost.  
**Cons:** You manage OS, updates, and Nginx/SSL.

---

## Checklist for any deployment

- [ ] Set **OPENAI_API_KEY** in the environment (never commit it).
- [ ] Use **HTTPS** in production (PaaS provides it; on VPS use Nginx/Caddy).
- [ ] If you use a **custom domain**, point DNS to the PaaS or to your VPS IP and configure the host there.

---

## Recommendation

- **Fastest path:** **Railway** or **Render** (Option 1 or 2). Connect repo, add `OPENAI_API_KEY`, deploy.
- **More control / production:** **Docker on a VPS** or **Google Cloud Run** (Option 3).
- **Cheapest long-term for always-on:** **Small VPS** (Option 4) with systemd + Nginx.
