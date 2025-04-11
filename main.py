import os, re, time, requests
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
with open("article.txt", "r", encoding="utf-8") as f: article = f.read()

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summary = summarizer(article, max_length=100, min_length=40, do_sample=False)[0]['summary_text']
key_points = summary.split('. ')[:5]

prompts = [f"Healthcare-themed illustration of: {pt}" for pt in key_points]
os.makedirs("output", exist_ok=True)

def get_image(prompt, idx):
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    r = requests.post(url, headers=headers, json={"inputs": prompt})
    if r.status_code == 200:
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt[:30]) + f"_{idx}_{int(time.time())}.png"
        with open(f"output/{name}", "wb") as f: f.write(r.content)
        print(f"✅ Saved: output/{name}")

for i, p in enumerate(prompts): get_image(p, i)
