# 🚀 Cloud Deployment Guide for Facial Emotion AI Web Application

This guide explains step-by-step how to deploy your **FastAPI & TensorFlow Facial Emotion AI Web Application** live on the cloud so anyone can access it via a public URL (`https://your-app.onrender.com`).

---

## 🛠️ Step 1: Create `requirements.txt` & `Dockerfile`

Ensure your project contains `python/requirements.txt`:

```text
fastapi>=0.110.0
uvicorn>=0.28.0
tensorflow>=2.16.0
fer>=22.5.0
opencv-python-headless>=4.9.0.80
numpy>=1.23.0
pydantic>=2.6.0
```

### Optional `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-get/lists/*

COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "python.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🌐 Option A: Deploy on Render.com (Recommended & Easiest)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Facial Emotion AI App"
   git remote add origin https://github.com/YOUR_USERNAME/EventRegistrationSystem.git
   git push -u origin main
   ```
2. **Sign up at [Render.com](https://render.com/)**.
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Configure deployment settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r python/requirements.txt`
   - **Start Command**: `uvicorn python.fastapi_server:app --host 0.0.0.0 --port $PORT`
6. Click **Create Web Service**.
7. Render will automatically build your app and give you a live HTTPS link! (e.g. `https://facial-emotion-ai.onrender.com`).

---

## 🤗 Option B: Deploy on Hugging Face Spaces (Free CPU & GPU)

1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Click **New Space** -> Choose **Docker** or **Streamlit/FastAPI** SDK.
3. Push your repository to the Hugging Face Git remote.
4. Hugging Face will host your app live with full camera access!
