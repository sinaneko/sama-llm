"""
LLM + RAG — نسخهٔ مخصوص Kaggle (برای تست)
سیستم پرسش‌وپاسخ فارسی روی اسناد با Qwen2.5 و FAISS

تفاوت‌ها با نسخهٔ اصلی (llm_script_final.py):
  • مدل و embedder مستقیماً از HuggingFace Hub دانلود می‌شوند (نه از پوشهٔ محلی models/)
  • تمام وابستگی به MySQL / تاریخچه / پنل ادمین حذف شده است
  • دو حالت تست:
        1) inline در نوت‌بوک  →  ask("سوال شما")
        2) سرور FastAPI + pyngrok  →  start_server()  (یک URL عمومی می‌دهد)

────────────────────────────────────────────────────────────────────
نصب وابستگی‌ها (در اولین سلول نوت‌بوک Kaggle اجرا کنید):

  !apt-get -qq update && apt-get -qq install -y poppler-utils tesseract-ocr tesseract-ocr-fas
  !pip install -q transformers accelerate sentence-transformers faiss-cpu
  !pip install -q pytesseract pdf2image pillow
  !pip install -q fastapi uvicorn nest-asyncio pyngrok
  # برای حالت 4-bit (اختیاری، روی یک GPU T4):
  !pip install -q bitsandbytes

نکته‌ها:
  • Internet را در تنظیمات نوت‌بوک Kaggle روشن کنید (برای دانلود مدل و ngrok).
  • Accelerator را روی GPU بگذارید (T4 یا P100).
  • برای ngrok یک authtoken رایگان از dashboard.ngrok.com بگیرید و در
    متغیّر محیطی NGROK_AUTHTOKEN قرار دهید (یا پایین در start_server بدهید).
────────────────────────────────────────────────────────────────────
"""

# ===== Built-in =====
import os
import re
import time
import json
import asyncio
import shutil
from threading import Thread
from typing import List, Dict, Any
from datetime import datetime

# ===== ML / AI =====
import torch
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

# ===== OCR / Files =====
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# ===== Backend / API =====
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse


# ===========================================================
# پیکربندی (با متغیّر محیطی قابل‌تغییر)
# ===========================================================

# مدل اصلی. می‌توانید برای تست سریع‌تر روی Kaggle مدل کوچک‌تری بگذارید،
# مثلاً "Qwen/Qwen2.5-3B-Instruct" یا "Qwen/Qwen2.5-1.5B-Instruct".
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "heydariAI/persian-embeddings")

# روی یک GPU T4 (۱۵GB) مدل 7B در fp16 به‌سختی جا می‌شود؛ با 4-bit مطمئن‌تر است.
USE_4BIT = os.getenv("LLM_USE_4BIT", "0") == "1"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
UPLOAD_DIR = os.getenv("LLM_UPLOAD_DIR", "/kaggle/working/doc")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===========================================================
# بارگذاری مدل و Embedder (از HuggingFace Hub)
# ===========================================================

print(f"🔄 در حال بارگذاری embedder: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

print(f"🔄 در حال بارگذاری مدل: {MODEL_NAME}  (4bit={USE_4BIT})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

_model_kwargs: Dict[str, Any] = dict(device_map="auto")
if USE_4BIT:
    from transformers import BitsAndBytesConfig
    _model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
else:
    _model_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_model_kwargs)
print("✅ مدل و embedder آماده شدند")


# ===========================================================
# State سراسری
# ===========================================================

extracted_texts: List[str] = []
metadata: List[Dict[str, Any]] = []
doc_embs = None
index = None


# ===========================================================
# Chunking
# ===========================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into smaller overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            best_break = -1
            for sep in ['. ', '؟ ', '! ', '\n', '، ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > best_break and last_sep > chunk_size // 2:
                    best_break = last_sep

            if best_break > 0:
                end = start + best_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks


def chunk_text_by_pages(pages_text: List[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split page texts into chunks, preserving page information."""
    all_chunks = []
    for page_num, page_text in enumerate(pages_text, 1):
        if not page_text or not page_text.strip():
            continue
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk_idx, chunk in enumerate(page_chunks):
            all_chunks.append({'text': chunk, 'page': page_num, 'chunk_index': chunk_idx})
    return all_chunks


# ===========================================================
# مدیریت اسناد در پوشه
# ===========================================================

def get_documents_from_folder(folder_path: str = UPLOAD_DIR) -> List[Dict[str, Any]]:
    """Read file list from the upload folder."""
    documents = []
    if not os.path.exists(folder_path):
        return documents

    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    for idx, filename in enumerate(os.listdir(folder_path), 1):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue
        file_stat = os.stat(filepath)
        file_type = "application/pdf" if ext == '.pdf' else f"image/{ext[1:]}"
        documents.append({
            "id": idx,
            "filename": filename,
            "filepath": filepath,
            "file_size": file_stat.st_size,
            "file_type": file_type,
            "created_at": datetime.fromtimestamp(file_stat.st_ctime),
            "updated_at": datetime.fromtimestamp(file_stat.st_mtime),
            "description": ""
        })
    return documents


# ===========================================================
# OCR و استخراج متن
# ===========================================================

def clean_persian_text(text: str) -> str:
    """Clean and normalize Persian text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_text = ' '.join(lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()


def extract_text_from_pdf_by_pages(pdf_path: str, language: str = 'fas') -> List[str]:
    """Extract Persian text from PDF using OCR, page by page."""
    if not os.path.exists(pdf_path):
        print(f"⚠️  فایل پیدا نشد: {pdf_path}")
        return []
    try:
        print(f"🔄 در حال پردازش: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=300)
        pages_text = []
        for i, image in enumerate(images, 1):
            print(f"  📄 صفحه {i}/{len(images)}")
            text = pytesseract.image_to_string(image, lang=language)
            cleaned_text = clean_persian_text(text) if text.strip() else ""
            pages_text.append(cleaned_text)
        total_chars = sum(len(p) for p in pages_text)
        print(f"✅ استخراج کامل شد - {len(images)} صفحه - {total_chars} کاراکتر")
        return pages_text
    except Exception as e:
        print(f"❌ خطا در پردازش {pdf_path}: {str(e)}")
        return []


def extract_text_from_image(image_path: str, language: str = 'fas') -> str:
    """Extract Persian text from an image file."""
    if not os.path.exists(image_path):
        print(f"⚠️  فایل پیدا نشد: {image_path}")
        return ""
    try:
        print(f"🔄 در حال پردازش تصویر: {image_path}")
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language)
        print(f"✅ استخراج کامل شد - {len(text)} کاراکتر")
        return text.strip()
    except Exception as e:
        print(f"❌ خطا در پردازش {image_path}: {str(e)}")
        return ""


def extract_chunks_from_file(filepath: str, language: str = 'fas') -> List[Dict]:
    """Extract and chunk text from a file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        pages_text = extract_text_from_pdf_by_pages(filepath, language)
        if pages_text:
            return chunk_text_by_pages(pages_text)
        return []
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        text = extract_text_from_image(filepath, language)
        if text:
            cleaned = clean_persian_text(text)
            chunks = chunk_text(cleaned)
            return [{'text': c, 'page': 1, 'chunk_index': i} for i, c in enumerate(chunks)]
        return []
    else:
        print(f"⚠️  نوع فایل پشتیبانی نمی‌شود: {ext}")
        return []


# ===========================================================
# مدیریت ایندکس FAISS
# ===========================================================

def rebuild_index():
    """Rebuild FAISS index from all extracted chunks."""
    global doc_embs, index
    if not extracted_texts:
        print("⚠️  هیچ متنی برای ایندکس‌گذاری وجود ندارد")
        doc_embs = None
        index = None
        return
    print(f"🔄 در حال ایجاد embedding برای {len(extracted_texts)} chunk...")
    doc_embs = embedder.encode(extracted_texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"✅ Embedding ایجاد شد - شکل: {doc_embs.shape}")
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    print(f"✅ ایندکس FAISS ساخته شد با {index.ntotal} chunk")


def add_document_to_index(filepath: str, language: str = 'fas') -> bool:
    """Add a new document to the index with chunking."""
    global extracted_texts, metadata
    filename = os.path.basename(filepath)
    for m in metadata:
        if m['filename'] == filename:
            print(f"⚠️  فایل {filename} قبلاً اضافه شده است")
            return False
    chunks = extract_chunks_from_file(filepath, language)
    if not chunks:
        print(f"⚠️  متنی از {filename} استخراج نشد")
        return False
    file_stat = os.stat(filepath)
    for chunk_data in chunks:
        extracted_texts.append(chunk_data['text'])
        metadata.append({
            'filename': filename,
            'filepath': filepath,
            'page': chunk_data['page'],
            'chunk_index': chunk_data['chunk_index'],
            'created_at': datetime.fromtimestamp(file_stat.st_ctime),
            'text_length': len(chunk_data['text'])
        })
    rebuild_index()
    print(f"✅ سند {filename} با {len(chunks)} chunk اضافه شد")
    return True


def remove_document_from_index(filename: str) -> bool:
    """Remove all chunks of a document from the index."""
    global extracted_texts, metadata
    indices_to_remove = [i for i, m in enumerate(metadata) if m['filename'] == filename]
    if not indices_to_remove:
        print(f"⚠️  فایل {filename} در ایندکس پیدا نشد")
        return False
    for i in reversed(indices_to_remove):
        extracted_texts.pop(i)
        metadata.pop(i)
    rebuild_index()
    print(f"✅ سند {filename} با {len(indices_to_remove)} chunk حذف شد")
    return True


def clear_all_documents():
    """Clear all documents and reset the index."""
    global extracted_texts, metadata, doc_embs, index
    extracted_texts = []
    metadata = []
    doc_embs = None
    index = None
    print("✅ همه اسناد پاک شدند")


def load_existing_documents(language: str = 'fas'):
    """Load all documents from the upload folder."""
    global extracted_texts, metadata
    print("=" * 60)
    print("🚀 شروع بارگذاری اسناد موجود (با chunking)")
    print("=" * 60)
    documents = get_documents_from_folder()
    if not documents:
        print("⚠️  هیچ سندی در پوشه آپلود پیدا نشد")
        return
    total_chunks = 0
    for doc in documents:
        print(f"\n📁 سند {doc['id']}: {doc['filename']}")
        chunks = extract_chunks_from_file(doc['filepath'], language)
        if chunks:
            file_stat = os.stat(doc['filepath'])
            for chunk_data in chunks:
                extracted_texts.append(chunk_data['text'])
                metadata.append({
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'page': chunk_data['page'],
                    'chunk_index': chunk_data['chunk_index'],
                    'description': doc.get('description', ''),
                    'created_at': doc.get('created_at'),
                    'text_length': len(chunk_data['text'])
                })
            total_chunks += len(chunks)
            print(f"✅ {len(chunks)} chunk ایجاد شد")
    if extracted_texts:
        rebuild_index()
    print("\n" + "=" * 60)
    print(f"✨ بارگذاری کامل شد:  📄 {len(documents)} سند  |  📦 {total_chunks} chunk")
    print("=" * 60)


def get_document_stats() -> Dict:
    """Get document and chunk statistics."""
    unique_files = set(m['filename'] for m in metadata)
    return {
        'total_documents': len(unique_files),
        'total_chunks': len(extracted_texts),
        'documents': list(unique_files),
        'chunks_per_doc': {
            f: sum(1 for m in metadata if m['filename'] == f)
            for f in unique_files
        }
    }


# ===========================================================
# بازیابی RAG
# ===========================================================

def retrieve_context(query, top_k=5, threshold=0.3):
    """Retrieve relevant chunks from the index."""
    if index is None or index.ntotal == 0:
        return None, None, None, None
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, min(top_k, index.ntotal))
    contexts, scores, indices, source_info = [], [], [], []
    for score, idx in zip(D[0], I[0]):
        if idx >= 0 and score >= threshold:
            contexts.append(extracted_texts[idx])
            scores.append(score)
            indices.append(idx)
            source_info.append({
                'filename': metadata[idx]['filename'],
                'page': metadata[idx]['page'],
                'score': float(score)
            })
    if not contexts:
        return None, None, None, None
    return contexts, scores, indices, source_info


def format_sources(source_info: List[Dict]) -> str:
    """Format source information for display."""
    if not source_info:
        return ""
    sources_by_file = {}
    for s in source_info:
        sources_by_file.setdefault(s['filename'], set()).add(s['page'])
    parts = []
    for fname, pages in sources_by_file.items():
        sorted_pages = sorted(pages)
        if len(sorted_pages) == 1:
            parts.append(f"{fname} (صفحه {sorted_pages[0]})")
        else:
            parts.append(f"{fname} (صفحات {', '.join(map(str, sorted_pages))})")
    return " | ".join(parts)


# ===========================================================
# Prompt و تولید پاسخ
# ===========================================================

def build_prompt(question, contexts):
    if contexts:
        context_text = "\n".join([f"- {c}" for c in contexts])
        prompt = f"""
You are a helpful assistant.
Your task is to answer the user's question using ONLY the provided Information below.
You must answer strictly in Persian (Farsi).

Example:
Information:
- پایتخت ایران تهران است.
Question:
پایتخت ایران کجاست؟
Answer (in Persian):
تهران پایتخت ایران است.

Information:
{context_text}

Question:
{question}

Answer (in Persian):
"""
    else:
        prompt = ""
    return prompt


# ----- حالت ۱: تست inline در نوت‌بوک -----

def ask(question: str, top_k: int = 5, threshold: float = 0.3,
        max_new_tokens: int = 1024, show_sources: bool = True) -> str:
    """
    تست سادهٔ inline در نوت‌بوک Kaggle.
    یک سوال می‌گیرد، context را از ایندکس می‌گیرد و پاسخ کامل را برمی‌گرداند.
        answer = ask("سوال شما؟")
    """
    start = time.time()
    contexts, scores, indices, source_info = retrieve_context(question, top_k=top_k, threshold=threshold)
    if not contexts:
        return "اطلاعاتی در این مورد در داده‌ها موجود نیست."

    prompt = build_prompt(question, contexts)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=14,
        do_sample=True,
    )
    gen = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # توقف در صورت تکرار قالب prompt
    for stop in ("\nInformation:", "\nExample:"):
        if stop in gen:
            gen = gen.split(stop)[0]
    answer = gen.strip()

    if show_sources and source_info:
        answer += f"\n\n📚 منابع: {format_sources(source_info)}"
    print(f"⏱️ مدت زمان پاسخ‌دهی: {time.time() - start:.2f} ثانیه")
    return answer


# ===========================================================
# HTML پنل ادمین (خودکفا، بدون وابستگی خارجی)
# ===========================================================

_ADMIN_PANEL_HTML = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>پنل تست — LLM RAG (Kaggle)</title>
<style>
  :root { --bg:#f4f6fb; --card:#fff; --ink:#222; --muted:#6b7280; --pri:#0984e3; --line:#e5e7eb; }
  * { box-sizing:border-box; }
  body { font-family:Tahoma,Vazirmatn,sans-serif; margin:0; background:var(--bg); color:var(--ink); }
  header { background:#2d3436; color:#fff; padding:14px 20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; flex:1; }
  .badge { background:#00b894; color:#fff; border-radius:20px; padding:3px 12px; font-size:13px; }
  .wrap { padding:18px 20px 50px; max-width:1000px; margin:0 auto; }
  .card { background:var(--card); border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.08); padding:16px 18px; margin-bottom:18px; }
  .card h2 { margin:0 0 12px; font-size:16px; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  input[type=text], textarea { flex:1; padding:10px 12px; border:1px solid #cbd5e1; border-radius:9px; font:inherit; }
  textarea { width:100%; min-height:60px; resize:vertical; }
  button { padding:10px 16px; background:var(--pri); color:#fff; border:none; border-radius:9px; cursor:pointer; font:inherit; }
  button:hover { background:#0b76c9; }
  button.danger { background:#d63031; } button.danger:hover { background:#b71c1c; }
  button.sec { background:#636e72; } button.sec:hover { background:#4b5557; }
  table { width:100%; border-collapse:collapse; }
  th,td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:right; font-size:14px; vertical-align:top; }
  th { background:#eef2f7; }
  td.num { text-align:center; color:var(--muted); white-space:nowrap; }
  .pill { display:inline-block; padding:2px 9px; border-radius:12px; font-size:12px; }
  .pill.ok { background:#d8f5e3; color:#0a8f4f; } .pill.no { background:#fde2e1; color:#c0392b; }
  .ans { background:#f1f9ff; border:1px solid #d6ebff; border-radius:9px; padding:12px; margin-top:12px; white-space:pre-wrap; color:#00708a; min-height:24px; }
  .muted { color:var(--muted); font-size:13px; }
  .stat { display:flex; gap:22px; flex-wrap:wrap; }
  .stat b { font-size:20px; display:block; }
</style>
</head>
<body>
<header>
  <h1>🧪 پنل تست LLM + RAG</h1>
  <span class="badge" id="model">…</span>
  <button class="sec" onclick="refreshAll()">↻ به‌روزرسانی</button>
</header>
<div class="wrap">

  <div class="card">
    <h2>وضعیت</h2>
    <div class="stat">
      <div><b id="st-docs">0</b><span class="muted">سند</span></div>
      <div><b id="st-chunks">0</b><span class="muted">chunk</span></div>
      <div><b id="st-ready">—</b><span class="muted">آماده؟</span></div>
      <div><b id="st-chat">0</b><span class="muted">گفتگو</span></div>
    </div>
  </div>

  <div class="card">
    <h2>پرسش از مدل</h2>
    <textarea id="q" placeholder="سوال خود را اینجا بنویسید…"></textarea>
    <div class="row" style="margin-top:10px">
      <button onclick="ask()">ارسال</button>
      <span class="muted" id="q-time"></span>
    </div>
    <div class="ans" id="ans"></div>
  </div>

  <div class="card">
    <h2>فایل‌ها</h2>
    <div class="row">
      <input type="file" id="file">
      <button onclick="uploadFile()">آپلود و ایندکس</button>
      <button class="sec" onclick="reindex()">ایندکس مجدد</button>
    </div>
    <div id="files" style="margin-top:12px"></div>
  </div>

  <div class="card">
    <h2>تاریخچهٔ گفتگو (درون‌حافظه)</h2>
    <div id="history"></div>
  </div>

</div>
<script>
async function jget(u){ const r=await fetch(u); return r.json(); }
async function refreshStatus(){
  const s=await jget('/');
  document.getElementById('model').textContent=s.model;
  document.getElementById('st-docs').textContent=s.total_documents;
  document.getElementById('st-chunks').textContent=s.total_chunks;
  document.getElementById('st-ready').textContent=s.index_ready?'✅':'❌';
}
async function refreshFiles(){
  const d=await jget('/admin/files');
  if(!d.items.length){ document.getElementById('files').innerHTML='<p class="muted">هیچ فایلی نیست. یک PDF/عکس آپلود کنید.</p>'; return; }
  let h='<table><tr><th>#</th><th>نام فایل</th><th>نوع</th><th>chunk</th><th>وضعیت</th><th></th></tr>';
  d.items.forEach((f,i)=>{ h+=`<tr><td class="num">${i+1}</td><td>${f.filename}</td><td class="muted">${f.file_type}</td><td class="num">${f.chunk_count}</td>`+
    `<td><span class="pill ${f.indexed?'ok':'no'}">${f.indexed?'ایندکس‌شده':'بدون متن'}</span></td>`+
    `<td><button class="danger" onclick="delFile('${f.filename}')">حذف</button></td></tr>`; });
  document.getElementById('files').innerHTML=h+'</table>';
}
async function refreshHistory(){
  const d=await jget('/admin/history');
  document.getElementById('st-chat').textContent=d.count;
  if(!d.items.length){ document.getElementById('history').innerHTML='<p class="muted">هنوز گفتگویی ثبت نشده.</p>'; return; }
  let h='<table><tr><th>#</th><th>سوال</th><th>پاسخ</th><th>زمان</th></tr>';
  d.items.forEach(c=>{ h+=`<tr><td class="num">${c.id}</td><td>${esc(c.question)}</td><td style="color:#00708a">${esc(c.answer)}</td><td class="muted">${c.created_at}</td></tr>`; });
  document.getElementById('history').innerHTML=h+'</table>';
}
function esc(s){ return (s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }
async function refreshAll(){ await refreshStatus(); await refreshFiles(); await refreshHistory(); }
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const a=document.getElementById('ans'); a.textContent='⏳ در حال پاسخ‌دهی…';
  const t0=performance.now();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const d=await r.json(); a.textContent=d.answer||d.error||'(بدون پاسخ)';
    document.getElementById('q-time').textContent=((performance.now()-t0)/1000).toFixed(1)+' ثانیه';
    refreshHistory();
  }catch(e){ a.textContent='خطا: '+e; }
}
async function uploadFile(){
  const f=document.getElementById('file').files[0]; if(!f){ alert('فایلی انتخاب نشده'); return; }
  const fd=new FormData(); fd.append('file',f);
  await fetch('/upload',{method:'POST',body:fd}); refreshAll();
}
async function delFile(name){
  if(!confirm('حذف '+name+'؟')) return;
  await fetch('/delete-file/'+encodeURIComponent(name),{method:'DELETE'}); refreshAll();
}
async function reindex(){ await fetch('/reindex',{method:'POST'}); refreshAll(); }
refreshAll();
</script>
</body>
</html>"""


# ===========================================================
# FastAPI App (حالت ۲: سرور + ngrok)
# ===========================================================

app = FastAPI(title="LLM RAG — Kaggle")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    stats = get_document_stats()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "index_ready": index is not None and index.ntotal > 0,
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.post("/upload")
async def upload_files(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    success = add_document_to_index(file_location)
    stats = get_document_stats()
    return {
        "filename": file.filename,
        "status": "uploaded_and_indexed" if success else "uploaded_but_not_indexed",
        "chunks_in_file": stats['chunks_per_doc'].get(file.filename, 0),
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.get("/uploaded-files")
async def list_files():
    files = []
    if os.path.exists(UPLOAD_DIR):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    return {"filesList": files}


@app.delete("/delete-file/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    remove_document_from_index(filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    stats = get_document_stats()
    return {
        "status": "deleted",
        "filename": filename,
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.post("/reindex")
async def reindex_all():
    clear_all_documents()
    load_existing_documents()
    stats = get_document_stats()
    return {
        "status": "reindexed",
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


async def stream_generator(question: str):
    loop = asyncio.get_running_loop()
    contexts, scores, indices, source_info = await loop.run_in_executor(
        None, retrieve_context, question, 5, 0.3
    )

    sources = []
    if source_info:
        best = source_info[0]
        sources = [{'filename': best['filename'], 'pages': [best['page']]}]
    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

    if not contexts:
        yield f"data: {json.dumps({'type': 'token', 'content': 'اطلاعاتی مرتبط پیدا نشد. لطفاً ابتدا فایل‌های خود را آپلود کنید.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    prompt = build_prompt(question, contexts)
    inputs = await loop.run_in_executor(
        None, lambda: tokenizer(prompt, return_tensors="pt").to(model.device)
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=14,
        do_sample=True,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    _SENTINEL = object()
    iterator = iter(streamer)
    generated_text = ""
    while True:
        new_text = await loop.run_in_executor(None, next, iterator, _SENTINEL)
        if new_text is _SENTINEL:
            break
        generated_text += new_text
        if "\nInformation:" in generated_text or "\nExample:" in generated_text:
            break
        yield f"data: {json.dumps({'type': 'token', 'content': new_text})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get('question')
    return StreamingResponse(stream_generator(question), media_type="text/event-stream")


# تاریخچهٔ گفتگو در حافظه (بدون دیتابیس) — فقط برای نمایش در پنل تست
CHAT_LOG: List[Dict[str, Any]] = []


@app.post("/ask")
async def ask_endpoint(request: Request):
    """
    Endpoint غیراستریم برای تست ساده.
    ورودی JSON: {"question": "<str>"}  →  خروجی: {"question", "answer"}
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "بدنهٔ درخواست JSON معتبر نیست"})

    question = (data.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=422, content={"error": "فیلد question الزامی است"})

    # از همان مسیر inline استفاده می‌کنیم (روی threadpool تا event loop بلاک نشود)
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, lambda: ask(question, show_sources=True))

    CHAT_LOG.append({
        "id": len(CHAT_LOG) + 1,
        "question": question,
        "answer": answer,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return {"question": question, "answer": answer}


# ===========================================================
# پنل ادمین (سبک، بدون login و بدون دیتابیس — مخصوص تست Kaggle)
# ===========================================================

@app.get("/admin/files")
async def admin_files():
    """لیست فایل‌های ایندکس‌شده به‌همراه تعداد chunk هر فایل."""
    stats = get_document_stats()
    items = [{
        "filename": d["filename"],
        "file_size": d["file_size"],
        "file_type": d["file_type"],
        "chunk_count": stats["chunks_per_doc"].get(d["filename"], 0),
        "indexed": stats["chunks_per_doc"].get(d["filename"], 0) > 0,
    } for d in get_documents_from_folder()]
    return {"count": len(items), "items": items,
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"]}


@app.get("/admin/history")
async def admin_history(limit: int = 200):
    """تاریخچهٔ گفتگوهای انجام‌شده از طریق /ask یا پنل (درون‌حافظه)."""
    items = list(reversed(CHAT_LOG))[:max(1, min(limit, 1000))]
    return {"count": len(items), "items": items}


@app.get("/panel", response_class=HTMLResponse)
async def panel():
    """پنل ادمین خودکفا (HTML/JS خالص، بدون وابستگی خارجی)."""
    return HTMLResponse(content=_ADMIN_PANEL_HTML)


# ===========================================================
# راه‌اندازی سرور روی Kaggle با pyngrok
# ===========================================================

_server_thread = None  # نگه‌داری رفرنس سرور برای جلوگیری از اجرای دوبارهٔ همزمان


def start_server(port: int = 8000, ngrok_token: str = None, wait: float = 2.0):
    """
    سرور FastAPI را روی Kaggle بالا می‌آورد و یک URL عمومی با ngrok می‌دهد.

        start_server(ngrok_token="توکن_شما")   # یا متغیّر محیطی NGROK_AUTHTOKEN

    نکته: uvicorn در یک ترد جدا اجرا می‌شود (نه با uvicorn.run)، چون نسخه‌های جدید
    uvicorn یک event loop تازه می‌سازند و داخل event loop در حال اجرای نوت‌بوک خطای
    «asyncio.run() cannot be called from a running event loop» می‌دهند. داخل ترد،
    event loopِ در حال اجرایی وجود ندارد و سرور بدون مشکل بالا می‌آید. این سلول هم
    non-blocking است؛ بعد از اجرا، سلول‌های دیگر را می‌توان اجرا کرد.
    """
    global _server_thread
    import uvicorn
    from pyngrok import ngrok

    token = ngrok_token or os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    # اگر سرور قبلاً بالا آمده، دوباره راه نمی‌اندازیم (پورت اشغال است)
    if _server_thread is not None and _server_thread.is_alive():
        print("ℹ️  سرور از قبل در حال اجراست. تونل‌های فعلی:")
        for t in ngrok.get_tunnels():
            print(f"   🌍 {t.public_url}")
        return _server_thread

    # راه‌اندازی uvicorn در ترد جدا
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    _server_thread = Thread(target=server.run, daemon=True)
    _server_thread.start()

    # کمی صبر تا سرور واقعاً به پورت گوش بدهد، بعد تونل ngrok را وصل کنیم
    time.sleep(wait)

    # بستن تونل‌های قبلی (در صورت اجرای مجدد سلول)
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)

    public_url = ngrok.connect(port).public_url
    print("=" * 60)
    print(f"🌍 URL عمومی: {public_url}")
    print(f"   • پنل ادمین:   {public_url}/panel   👈 این را در مرورگر باز کن")
    print(f"   • تست سلامت:   GET  {public_url}/")
    print(f"   • پرسش:        POST {public_url}/ask     body: {{\"question\": \"...\"}}")
    print(f"   • آپلود فایل:  POST {public_url}/upload  (multipart file)")
    print("=" * 60)
    print("✅ سرور در پس‌زمینه اجرا می‌شود (سلول مسدود نمی‌شود).")
    return _server_thread


# ===========================================================
# نقطهٔ ورود
# ===========================================================
#
# روی Kaggle معمولاً این فایل را به‌صورت سلول‌به‌سلول اجرا می‌کنید، نه با python.
# دو راه تست:
#
#   ── حالت ۱: inline ───────────────────────────────────────
#   load_existing_documents(language='fas')   # بعد از قرار دادن PDF/عکس در UPLOAD_DIR
#   print(ask("سوال شما؟"))
#
#   ── حالت ۲: سرور + ngrok ─────────────────────────────────
#   load_existing_documents(language='fas')
#   start_server(ngrok_token="توکن_ngrok_شما")
#
# اگر فایلی برای ایندکس ندارید، می‌توانید سریع یک متن تستی اضافه کنید:
#   extracted_texts.append("پایتخت ایران تهران است.")
#   metadata.append({"filename": "test.txt", "page": 1, "chunk_index": 0})
#   rebuild_index()
#   print(ask("پایتخت ایران کجاست؟"))
# ===========================================================

if __name__ == "__main__":
    load_existing_documents(language='fas')
    if index is not None:
        stats = get_document_stats()
        print(f"✅ ایندکس آماده: 📄 {stats['total_documents']} سند | 📦 {stats['total_chunks']} chunk")
    else:
        print("⚠️  هنوز سندی ایندکس نشده — فایل‌ها را در UPLOAD_DIR بگذارید یا متن تستی اضافه کنید")
    start_server()
