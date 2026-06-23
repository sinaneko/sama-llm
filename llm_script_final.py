"""
LLM + RAG Server with FastAPI
Persian document question-answering system using Qwen2.5-7B and FAISS
"""

# ===== Installation (run once before starting) =====
# LLM روی سرویس Ollama اجرا می‌شود (جدا)؛ این اسکریپت فقط به آن وصل می‌شود.
#   - نصب Ollama و مدل:  ollama pull qwen2.5:7b   (روی ماشین دارای GPU انویدیا)
# pip install ollama gradio websockets fastapi uvicorn
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch CPU برای embedder
# pip install transformers faiss-cpu sentence-transformers
# pip install opencv-python numpy pytesseract pdf2image
# pip install opencv-python-headless arabic-reshaper pdfplumber python-bidi
# pip install mysql-connector-python
# apt-get install poppler-utils tesseract-ocr tesseract-ocr-fas

# ===== Built-in =====
import os
import re
import time
import json
import asyncio
import shutil
import secrets
from typing import List, Dict, Any
from datetime import datetime

# ===== ML / AI =====
import faiss
import ollama  # کلاینت LLM؛ مدل qwen2.5:7b روی سرویس Ollama (GPU) اجرا می‌شود
from sentence_transformers import SentenceTransformer

# ===== OCR / Files =====
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# ===== Backend / API =====
from fastapi import FastAPI, Request, UploadFile, File, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uvicorn
import gradio as gr

# ===== Database =====
import mysql.connector


# ===========================================================
# Model & Embedder Initialization
# ===========================================================

# Embedder روی CPU اجرا می‌شود تا این کانتینر اصلاً به GPU/torch مخصوص Blackwell نیاز نداشته باشد.
# مدل embedding کوچک است و روی CPU به‌اندازهٔ کافی سریع است.
embedder = SentenceTransformer("heydariAI/persian-embeddings", device="cpu")

# LLM روی سرویس جدا Ollama (با GPU) اجرا می‌شود. این کانتینر فقط از طریق شبکه به آن وصل می‌شود.
# Ollama روی کارت‌های Blackwell به‌صورت خودکار کار می‌کند و خودش CUDA را مدیریت می‌کند.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# پارامترهای تولید (معادل تنظیمات قبلی transformers روی Ollama).
OLLAMA_OPTIONS = {
    "num_predict": 1024,   # = max_new_tokens
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 50,
    "repeat_penalty": 1.0,
}


def _wait_for_ollama(retries: int = 30, delay: float = 2.0) -> bool:
    """منتظر آماده‌شدن سرویس Ollama می‌ماند (مثلاً وقتی هم‌زمان با اپ بالا می‌آید)."""
    for attempt in range(1, retries + 1):
        try:
            ollama_client.list()
            return True
        except Exception:
            if attempt == 1:
                print("⏳ در انتظار آماده‌شدن سرویس Ollama...")
            time.sleep(delay)
    return False


def ensure_ollama_model(model_tag: str = None) -> None:
    """در صورت نبودِ مدل روی سرویس Ollama، آن را یک‌بار دانلود (pull) می‌کند.
    ابتدا منتظر آماده‌شدن سرویس می‌ماند تا در شرایط رقابتی استارت، مدل از قلم نیفتد."""
    model_tag = model_tag or OLLAMA_MODEL

    if not _wait_for_ollama():
        print("⚠️  سرویس Ollama در دسترس نشد؛ مدل بعداً هنگام اولین درخواست تلاش می‌شود.")
        return

    try:
        local = {m.get("model") for m in ollama_client.list().get("models", [])}
        if model_tag in local or any(str(t).startswith(model_tag) for t in local):
            print(f"✅ مدل Ollama موجود است: {model_tag}")
            return
        print(f"🔄 در حال دانلود مدل Ollama: {model_tag} (بار اول کمی طول می‌کشد)...")
        ollama_client.pull(model_tag)
        print(f"✅ دانلود مدل Ollama کامل شد: {model_tag}")
    except Exception as e:
        print(f"⚠️  بررسی/دانلود مدل Ollama ناموفق بود (هنگام درخواست دوباره تلاش می‌شود): {e}")


# ===========================================================
# Global State
# ===========================================================

extracted_texts: List[str] = []  # Stores chunks, not full documents
metadata: List[Dict[str, Any]] = []  # Metadata for each chunk
doc_embs = None
index = None

CHUNK_SIZE = 512       # Number of characters per chunk
CHUNK_OVERLAP = 100    # Overlap between chunks to preserve context

UPLOAD_DIR = "doc"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===========================================================
# Text Chunking
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
            all_chunks.append({
                'text': chunk,
                'page': page_num,
                'chunk_index': chunk_idx
            })

    return all_chunks


# ===========================================================
# Database (for future use)
# ===========================================================

def get_db_connection():
    """Connect to MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password",
        database="documents_db"
    )


def get_documents_from_db() -> List[Dict[str, Any]]:
    """Read document list from database (for future use)."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT id, filename, filepath, file_size, file_type,
               created_at, updated_at, description
        FROM documents
        ORDER BY created_at DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    return results


# ===========================================================
# Document Management
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
# OCR & Text Extraction
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


def extract_text_from_pdf(pdf_path: str, language: str = 'fas') -> str:
    """Extract text from PDF (compatibility wrapper)."""
    pages = extract_text_from_pdf_by_pages(pdf_path, language)
    return "\n\n".join(pages)


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


def extract_text_from_file(filepath: str, language: str = 'fas') -> str:
    """Extract text from file (compatibility wrapper)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        text = extract_text_from_pdf(filepath, language)
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        text = extract_text_from_image(filepath, language)
    else:
        print(f"⚠️  نوع فایل پشتیبانی نمی‌شود: {ext}")
        return ""

    return clean_persian_text(text) if text else ""


# ===========================================================
# FAISS Index Management
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
    print(f"✨ بارگذاری کامل شد:")
    print(f"   📄 {len(documents)} سند")
    print(f"   📦 {total_chunks} chunk")
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
# RAG Retrieval
# ===========================================================

def retrieve_context(query, top_k=5, threshold=0.3):
    """Retrieve relevant chunks from the index."""
    if index is None or index.ntotal == 0:
        return None, None, None, None

    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, min(top_k, index.ntotal))

    contexts = []
    scores = []
    indices = []
    source_info = []

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
        fname = s['filename']
        if fname not in sources_by_file:
            sources_by_file[fname] = set()
        sources_by_file[fname].add(s['page'])

    parts = []
    for fname, pages in sources_by_file.items():
        sorted_pages = sorted(pages)
        if len(sorted_pages) == 1:
            parts.append(f"{fname} (صفحه {sorted_pages[0]})")
        else:
            parts.append(f"{fname} (صفحات {', '.join(map(str, sorted_pages))})")

    return " | ".join(parts)


def debug_search(query: str, top_k: int = 5) -> None:
    """Print detailed search results for debugging."""
    print(f"\n{'='*60}")
    print(f"🔍 سوال: {query}")
    print(f"{'='*60}\n")

    contexts, scores, indices, source_info = retrieve_context(query, top_k=top_k, threshold=0.0)

    if not contexts:
        print("❌ هیچ نتیجه‌ای پیدا نشد!")
        return

    print(f"📊 تعداد نتایج: {len(contexts)}\n")

    for i, (ctx, score, idx, src) in enumerate(zip(contexts, scores, indices, source_info)):
        print(f"--- نتیجه {i+1} ---")
        print(f"📁 فایل: {src['filename']}")
        print(f"📄 صفحه: {src['page']}")
        print(f"📈 امتیاز: {score:.4f}")
        print(f"🔢 ایندکس در metadata: {idx}")
        print(f"📝 متن (100 کاراکتر اول):")
        print(f"   {ctx[:100]}...")
        print()

    print(f"{'='*60}\n")


def show_all_chunks():
    """Print all chunks and their metadata."""
    print(f"\n{'='*60}")
    print(f"📦 لیست همه chunks در ایندکس")
    print(f"{'='*60}\n")

    if not metadata:
        print("❌ هیچ chunk ای در ایندکس وجود ندارد!")
        return

    for i, (text, meta) in enumerate(zip(extracted_texts, metadata)):
        print(f"[{i}] 📁 {meta['filename']} | 📄 صفحه {meta['page']} | 📊 chunk {meta['chunk_index']}")
        print(f"    📝 {text[:80]}...")
        print()


# ===========================================================
# Prompt & Generation
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


def ask_with_rag_gradio(question, history):
    start = time.time()

    contexts, scores, indices, source_info = retrieve_context(question, top_k=5, threshold=0.3)

    if not contexts:
        yield "اطلاعاتی در این مورد در داده‌ها موجود نیست."
        return

    prompt = build_prompt(question, contexts)

    generated_text = ""
    for chunk in ollama_client.generate(
        model=OLLAMA_MODEL, prompt=prompt, stream=True, options=OLLAMA_OPTIONS
    ):
        new_text = chunk.get("response", "")
        if not new_text:
            continue
        generated_text += new_text

        if "\nInformation:" in generated_text:
            yield generated_text.split("\nInformation:")[0].strip()
            return

        if "\nExample:" in generated_text:
            yield generated_text.split("\nExample:")[0].strip()
            return

        yield generated_text

    if source_info:
        sources_text = format_sources(source_info)
        yield generated_text + f"\n\n📚 منابع: {sources_text}"

    elapsed = time.time() - start
    print(f"⏱️ مدت زمان پاسخ‌دهی: {elapsed:.2f} ثانیه")


# ===========================================================
# FastAPI App
# ===========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_files(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    success = add_document_to_index(file_location)
    stats = get_document_stats()
    chunk_count = stats['chunks_per_doc'].get(file.filename, 0)

    _safe_db(
        db_upsert_document,
        file.filename, file_location, os.path.getsize(file_location),
        _guess_file_type(file.filename),
        "indexed" if success else "not_indexed", chunk_count,
        None if success else "متنی از فایل استخراج نشد",
    )

    if success:
        return {
            "filename": file.filename,
            "status": "uploaded_and_indexed",
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks'],
            "chunks_in_file": chunk_count
        }
    else:
        return {
            "filename": file.filename,
            "status": "uploaded_but_not_indexed",
            "message": "فایل ذخیره شد اما متنی استخراج نشد"
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
    _safe_db(db_delete_document, filename)
    stats = get_document_stats()
    return {
        "status": "deleted",
        "filename": filename,
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks']
    }


@app.delete("/clear-files")
async def clear_files():
    clear_all_documents()
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return {"status": "cleared", "total_documents": 0, "total_chunks": 0}


@app.post("/reindex")
async def reindex_all():
    clear_all_documents()
    load_existing_documents()
    stats = get_document_stats()
    return {
        "status": "reindexed",
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks']
    }


@app.get("/index-status")
async def index_status():
    stats = get_document_stats()
    return {
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
        "index_ready": index is not None and index.ntotal > 0,
        "chunks_per_doc": stats['chunks_per_doc'],
        "documents": stats['documents']
    }


@app.post("/debug-search")
async def debug_search_api(request: Request):
    data = await request.json()
    query = data.get('question', '')
    top_k = data.get('top_k', 5)
    if not query:
        return {"error": "سوال خالی است"}
    contexts, scores, indices, source_info = retrieve_context(query, top_k=top_k, threshold=0.0)
    if not contexts:
        return {"query": query, "results": [], "message": "هیچ نتیجه‌ای پیدا نشد"}
    results = []
    for i, (ctx, score, idx, src) in enumerate(zip(contexts, scores, indices, source_info)):
        results.append({
            "rank": i + 1,
            "filename": src['filename'],
            "page": src['page'],
            "score": round(float(score), 4),
            "index_in_metadata": int(idx),
            "text_preview": ctx[:200] + "..." if len(ctx) > 200 else ctx,
            "full_text": ctx
        })
    return {"query": query, "total_results": len(results), "results": results}


@app.get("/all-chunks")
async def get_all_chunks():
    if not metadata:
        return {"chunks": [], "message": "هیچ chunk ای وجود ندارد"}
    chunks = []
    for i, (text, meta) in enumerate(zip(extracted_texts, metadata)):
        chunks.append({
            "index": i,
            "filename": meta['filename'],
            "page": meta['page'],
            "chunk_index": meta['chunk_index'],
            "text_preview": text[:150] + "..." if len(text) > 150 else text
        })
    return {"total_chunks": len(chunks), "chunks": chunks}


async def stream_generator(question: str):
    loop = asyncio.get_running_loop()

    # بازیابی context (embedder.encode + جستجوی FAISS) کار سنگین CPU/GPU است؛
    # روی threadpool اجرا می‌شود تا event loop بلاک نشود و بقیهٔ APIها پاسخگو بمانند.
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

    # استریم پاسخ از Ollama. کلاینت Ollama سینکرون است و هر next() تا آماده‌شدن توکن
    # بعدی بلاک می‌شود؛ پس مثل قبل هر next() را روی threadpool اجرا می‌کنیم تا event loop
    # بین توکن‌ها آزاد بماند و بقیهٔ APIها پاسخگو باشند.
    def _start_stream():
        return ollama_client.generate(
            model=OLLAMA_MODEL, prompt=prompt, stream=True, options=OLLAMA_OPTIONS
        )

    iterator = await loop.run_in_executor(None, _start_stream)

    _SENTINEL = object()
    generated_text = ""
    while True:
        chunk = await loop.run_in_executor(None, next, iterator, _SENTINEL)
        if chunk is _SENTINEL:
            break
        new_text = chunk.get("response", "")
        if not new_text:
            if chunk.get("done"):
                break
            continue
        generated_text += new_text
        if "\nInformation:" in generated_text:
            break
        if "\nExample:" in generated_text:
            break
        yield f"data: {json.dumps({'type': 'token', 'content': new_text})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get('question')
    return StreamingResponse(stream_generator(question), media_type="text/event-stream")


# ===========================================================
# Gradio UI
# ===========================================================

chat_ui = gr.ChatInterface(
    fn=ask_with_rag_gradio,
    title="Chat with LLM + RAG (Chunked)",
    description="چت با مدل LLM و بازیابی اطلاعات از اسناد چند صفحه‌ای (استریم فعال)"
)


# ===========================================================
# LLM Bot API  +  Chat History Logging
# -----------------------------------------------------------
# این سلول یک Endpoint جدید (POST /ask) به سرور FastAPI اضافه می‌کند که:
#   1) درخواست کاربر (user_id و question) را دریافت می‌کند،
#   2) فعلاً بدون توجه به محتوای سؤال، همیشه پاسخ "سلام" برمی‌گرداند،
#   3) هر گفتگو را در جدول llm_chat_history ذخیره می‌کند
#      (شناسه کاربر، شماره ترتیبی سؤال، متن سؤال، پاسخ، زمان ثبت).
# نکته: این سلول به متغیّرهای سراسری cell قبلی (app, Request, JSONResponse,
#       mysql, os) متّکی است؛ پس باید بعد از سلول اصلی اجرا شود.
# ===========================================================

# ----- تنظیمات اتصال به دیتابیس (قابل‌تنظیم با متغیّر محیطی) -----
# پیش‌فرض‌ها مطابق دیتابیس teamgram است؛ هنگام اجرا در محیط دیگر فقط
# متغیّرهای محیطی LLM_DB_* را تنظیم کنید.
LLM_DB_CONFIG = {
    "host":     os.getenv("LLM_DB_HOST", "127.0.0.1"),
    "port":     int(os.getenv("LLM_DB_PORT", "3306")),
    "user":     os.getenv("LLM_DB_USER", "teamgram"),
    "password": os.getenv("LLM_DB_PASSWORD", "teamgram"),
    "database": os.getenv("LLM_DB_NAME", "teamgram"),
}


def get_llm_db_connection():
    """اتصال به دیتابیسی که تاریخچهٔ گفتگوی ربات LLM در آن ذخیره می‌شود."""
    return mysql.connector.connect(**LLM_DB_CONFIG)


def init_llm_history_table() -> None:
    """
    ایجاد جدول تاریخچه در صورت عدم وجود (به‌عنوان شبکهٔ ایمنی در کنار migration).
    این کار idempotent است و در صورت وجود جدول کاری انجام نمی‌دهد.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS llm_chat_history (
        id              BIGINT       NOT NULL AUTO_INCREMENT,
        user_id         BIGINT       NOT NULL,
        question_number INT          NOT NULL,
        question        TEXT         NOT NULL,
        answer          TEXT         NOT NULL,
        created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_user_id (user_id),
        UNIQUE KEY uq_user_question (user_id, question_number)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def save_chat_history(user_id: int, question: str, answer: str):
    """
    ذخیرهٔ یک گفتگو در جدول تاریخچه و برگرداندن (شماره سؤال، زمان ثبت).
    شماره سؤال برای هر کاربر (هر گفتگو) به‌صورت ترتیبی افزایش می‌یابد.
    برای جلوگیری از تداخل در شرایط رقابتی از تراکنش و قفل SELECT ... FOR UPDATE
    استفاده شده و در صورت برخورد با کلید یکتا یک‌بار دوباره تلاش می‌شود.
    """
    last_err = None
    for _attempt in range(3):  # تلاش مجدد محدود در صورت رقابت روی شماره سؤال
        conn = get_llm_db_connection()
        try:
            cur = conn.cursor()
            conn.start_transaction()

            # محاسبهٔ شمارهٔ سؤال بعدی برای همین کاربر
            cur.execute(
                "SELECT COALESCE(MAX(question_number), 0) + 1 "
                "FROM llm_chat_history WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            question_number = int(cur.fetchone()[0])

            # درج رکورد تاریخچه
            cur.execute(
                "INSERT INTO llm_chat_history "
                "(user_id, question_number, question, answer) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, question_number, question, answer),
            )
            conn.commit()
            cur.close()
            return question_number, datetime.utcnow().isoformat()
        except mysql.connector.IntegrityError as e:
            # شماره سؤال هم‌زمان توسط درخواست دیگری گرفته شده؛ دوباره تلاش کن
            conn.rollback()
            last_err = e
        finally:
            conn.close()
    raise last_err if last_err else RuntimeError("ذخیرهٔ تاریخچه ناموفق بود")


async def generate_llm_answer(question: str) -> str:
    """
    تولید پاسخ کامل با مصرف همان مسیر استریم (stream_generator).
    رویدادهای SSE را می‌خواند، توکن‌های نوع «token» را به هم می‌چسباند و
    متن نهایی را برمی‌گرداند. این‌طور منطق تولید پاسخ یک‌جا متمرکز می‌ماند.
    """
    answer = ""
    async for chunk in stream_generator(question):
        # هر chunk به شکل «data: {json}\n\n» است
        line = chunk.strip()
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[len("data:"):].strip())
        except Exception:
            continue
        if obj.get("type") == "token":
            answer += obj.get("content", "")
        elif obj.get("type") == "done":
            break

    answer = answer.strip()
    return answer if answer else "پاسخی تولید نشد."


def fetch_chat_history(user_id: int = None, limit: int = 200):
    """
    خواندن تاریخچهٔ گفتگوها برای نمایش در پنل ادمین.
    اگر user_id داده شود فقط گفتگوهای همان کاربر، در غیر این صورت همه.
    خروجی: لیستی از دیکشنری‌ها (جدیدترین اول).
    """
    limit = max(1, min(int(limit), 1000))  # محدودیت معقول برای جلوگیری از بار زیاد
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        if user_id is not None:
            cur.execute(
                "SELECT id, user_id, question_number, question, answer, created_at "
                "FROM llm_chat_history WHERE user_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (int(user_id), limit),
            )
        else:
            cur.execute(
                "SELECT id, user_id, question_number, question, answer, created_at "
                "FROM llm_chat_history ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()
        cur.close()
        # تبدیل datetime به رشته برای سریال‌سازی JSON
        for r in rows:
            if r.get("created_at") is not None:
                r["created_at"] = str(r["created_at"])
        return rows
    finally:
        conn.close()


# ===========================================================
# Admin Authentication (نام کاربری/رمز عبور پنل ادمین)
# ===========================================================
# نام کاربری و رمز عبور از متغیّرهای محیطی خوانده می‌شوند. حتماً در محیط واقعی
# ADMIN_USERNAME و ADMIN_PASSWORD را تنظیم کنید؛ مقادیر پیش‌فرض فقط برای تست‌اند.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")
ADMIN_TOKEN_TTL = int(os.getenv("ADMIN_TOKEN_TTL", str(12 * 3600)))  # مدت اعتبار توکن (ثانیه)

# نگه‌داری توکن‌های فعال در حافظه: token -> زمان انقضا (epoch seconds).
# با ری‌استارت سرور، کاربر باید دوباره وارد شود (برای پنل ادمین کافی است).
_ADMIN_TOKENS: Dict[str, float] = {}


def _issue_admin_token() -> str:
    """ساخت یک توکن تصادفی امن و ثبت آن با زمان انقضا."""
    token = secrets.token_urlsafe(32)
    _ADMIN_TOKENS[token] = time.time() + ADMIN_TOKEN_TTL
    return token


def _token_valid(token: str) -> bool:
    """بررسی معتبر بودن و منقضی‌نشدن توکن."""
    exp = _ADMIN_TOKENS.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        _ADMIN_TOKENS.pop(token, None)
        return False
    return True


def require_admin(authorization: str = Header(None)):
    """
    وابستگی FastAPI برای محافظت از مسیرهای ادمین.
    توکن باید در هدر «Authorization: Bearer <token>» ارسال شود.
    """
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not _token_valid(token):
        raise HTTPException(status_code=401, detail="نیاز به ورود ادمین")
    return token


# ===========================================================
# Documents DB (مدیریت فایل‌ها برای RAG)
# ===========================================================

def init_documents_table() -> None:
    """ایجاد جدول rag_documents در صورت عدم وجود (idempotent)."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS rag_documents (
        id           BIGINT        NOT NULL AUTO_INCREMENT,
        filename     VARCHAR(512)  NOT NULL,
        filepath     VARCHAR(1024) NOT NULL,
        file_size    BIGINT        NOT NULL DEFAULT 0,
        file_type    VARCHAR(128)  NOT NULL DEFAULT '',
        status       VARCHAR(32)   NOT NULL DEFAULT 'pending',
        chunk_count  INT           NOT NULL DEFAULT 0,
        error        TEXT          NULL,
        created_at   TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at   TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_filename (filename)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def db_upsert_document(filename, filepath, file_size, file_type, status, chunk_count, error=None):
    """درج یا به‌روزرسانی رکورد یک فایل در جدول rag_documents."""
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO rag_documents "
            "(filename, filepath, file_size, file_type, status, chunk_count, error) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "filepath=VALUES(filepath), file_size=VALUES(file_size), "
            "file_type=VALUES(file_type), status=VALUES(status), "
            "chunk_count=VALUES(chunk_count), error=VALUES(error)",
            (filename, filepath, int(file_size), file_type, status, int(chunk_count), error),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def db_delete_document(filename):
    """حذف رکورد یک فایل از جدول rag_documents."""
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM rag_documents WHERE filename = %s", (filename,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def db_list_documents():
    """خواندن لیست فایل‌ها از جدول rag_documents (جدیدترین اول)."""
    conn = get_llm_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, filename, filepath, file_size, file_type, status, "
            "chunk_count, error, created_at, updated_at "
            "FROM rag_documents ORDER BY id DESC"
        )
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            for k in ("created_at", "updated_at"):
                if r.get(k) is not None:
                    r[k] = str(r[k])
        return rows
    finally:
        conn.close()


def _safe_db(fn, *args, **kwargs):
    """اجرای یک عملیات دیتابیس به‌صورت best-effort؛ خطا فقط لاگ می‌شود و ایندکس را متوقف نمی‌کند."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  عملیات دیتابیس اسناد ناموفق بود: {e}")
        return None


def _guess_file_type(filename: str) -> str:
    """حدس نوع MIME ساده بر اساس پسوند فایل."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        return f"image/{ext[1:]}"
    return ""


def sync_documents_db():
    """
    هم‌گام‌سازی جدول rag_documents با وضعیت فعلی ایندکس و پوشهٔ آپلود.
    برای هر فایل موجود در پوشه، تعداد chunk های ایندکس‌شده و وضعیت ثبت می‌شود.
    """
    counts: Dict[str, int] = {}
    for m in metadata:
        counts[m['filename']] = counts.get(m['filename'], 0) + 1

    for d in get_documents_from_folder():
        cc = counts.get(d['filename'], 0)
        status = "indexed" if cc > 0 else "not_indexed"
        db_upsert_document(
            d['filename'], d['filepath'], d['file_size'], d['file_type'], status, cc, None
        )


# تلاش برای ساخت جدول‌ها هنگام بارگذاری ماژول (در صورت در دسترس نبودن DB، خطا نادیده گرفته می‌شود)
try:
    init_llm_history_table()
    init_documents_table()
    print("✅ جدول‌های llm_chat_history و rag_documents آماده‌اند")
except Exception as _e:
    print(f"⚠️  اتصال/ساخت جدول‌ها ممکن نشد (در زمان درخواست دوباره تلاش می‌شود): {_e}")


@app.post("/ask")
async def ask_endpoint(request: Request):
    """
    Endpoint اصلی ربات LLM.
    ورودی (JSON): {"user_id": <int>, "question": "<str>"}
    خروجی (JSON): {user_id, question_number, question, answer, created_at}
    """
    # ۱) خواندن و اعتبارسنجی بدنهٔ درخواست
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "بدنهٔ درخواست JSON معتبر نیست"})

    user_id = data.get("user_id")
    question = (data.get("question") or "").strip()

    if user_id is None or question == "":
        return JSONResponse(
            status_code=422,
            content={"error": "فیلدهای user_id و question الزامی هستند"},
        )

    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return JSONResponse(status_code=422, content={"error": "user_id باید عدد باشد"})

    # ۲) تولید پاسخ واقعی با RAG (از مسیر stream_generator)
    answer = await generate_llm_answer(question)

    # ۳) ذخیرهٔ تاریخچه؛ حتی اگر ذخیره ناموفق بود، پاسخ به کاربر برگردانده می‌شود
    try:
        question_number, created_at = save_chat_history(user_id, question, answer)
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "warning": f"پاسخ تولید شد اما ذخیرهٔ تاریخچه ناموفق بود: {e}",
            },
        )

    # ۴) پاسخ نهایی
    return {
        "user_id": user_id,
        "question_number": question_number,
        "question": question,
        "answer": answer,
        "created_at": created_at,
    }


# ===========================================================
# پنل ادمین مشاهدهٔ پیام‌ها
# ===========================================================

# مسیر فایل HTML پنل ادمین (کنار همین اسکریپت قرار دارد).
_PANEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin_panel.html")


@app.post("/admin/login")
async def admin_login(request: Request):
    """ورود ادمین. ورودی JSON: {username, password} → خروجی: {token, expires_in}."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "بدنهٔ درخواست JSON معتبر نیست"})

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user_ok = secrets.compare_digest(username, ADMIN_USERNAME)
    pass_ok = secrets.compare_digest(password, ADMIN_PASSWORD)
    if not (user_ok and pass_ok):
        return JSONResponse(status_code=401, content={"error": "نام کاربری یا رمز عبور نادرست است"})

    token = _issue_admin_token()
    return {"token": token, "expires_in": ADMIN_TOKEN_TTL}


@app.get("/history")
async def history_endpoint(user_id: int = None, limit: int = 200, _admin: str = Depends(require_admin)):
    """
    خروجی JSON تاریخچهٔ گفتگوها (محافظت‌شده با ورود ادمین).
    پارامترهای اختیاری: user_id (فیلتر کاربر)، limit (حداکثر تعداد ردیف).
    """
    try:
        rows = fetch_chat_history(user_id=user_id, limit=limit)
        return {"count": len(rows), "items": rows}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"اتصال به دیتابیس ممکن نشد: {e}"})


# ----- مدیریت فایل‌ها (محافظت‌شده با ورود ادمین) -----

@app.get("/admin/files")
async def admin_list_files(_admin: str = Depends(require_admin)):
    """لیست فایل‌های مدیریت‌شده. ترجیحاً از دیتابیس؛ در صورت در دسترس نبودن، از پوشه."""
    try:
        rows = db_list_documents()
        return {"source": "db", "count": len(rows), "items": rows}
    except Exception:
        # fallback: ساخت لیست از روی پوشه و وضعیت فعلی ایندکس
        counts: Dict[str, int] = {}
        for m in metadata:
            counts[m['filename']] = counts.get(m['filename'], 0) + 1
        items = [{
            "filename": d["filename"],
            "filepath": d["filepath"],
            "file_size": d["file_size"],
            "file_type": d["file_type"],
            "status": "indexed" if counts.get(d["filename"], 0) > 0 else "not_indexed",
            "chunk_count": counts.get(d["filename"], 0),
        } for d in get_documents_from_folder()]
        return {"source": "folder", "count": len(items), "items": items}


@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), _admin: str = Depends(require_admin)):
    """آپلود فایل جدید توسط ادمین، افزودن به ایندکس RAG و ثبت در دیتابیس."""
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    success = add_document_to_index(file_location)
    stats = get_document_stats()
    chunk_count = stats['chunks_per_doc'].get(file.filename, 0)

    file_size = os.path.getsize(file_location)
    _safe_db(
        db_upsert_document,
        file.filename, file_location, file_size, _guess_file_type(file.filename),
        "indexed" if success else "not_indexed", chunk_count,
        None if success else "متنی از فایل استخراج نشد",
    )

    return {
        "filename": file.filename,
        "status": "uploaded_and_indexed" if success else "uploaded_but_not_indexed",
        "chunks_in_file": chunk_count,
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.delete("/admin/files/{filename}")
async def admin_delete_file(filename: str, _admin: str = Depends(require_admin)):
    """حذف فایل از ایندکس RAG، پوشه و دیتابیس."""
    remove_document_from_index(filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    _safe_db(db_delete_document, filename)

    stats = get_document_stats()
    return {
        "status": "deleted",
        "filename": filename,
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.post("/admin/reindex")
async def admin_reindex(_admin: str = Depends(require_admin)):
    """ایندکس مجدد همهٔ فایل‌های پوشهٔ آپلود و هم‌گام‌سازی دیتابیس."""
    clear_all_documents()
    load_existing_documents()
    _safe_db(sync_documents_db)

    stats = get_document_stats()
    return {
        "status": "reindexed",
        "total_documents": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
    }


@app.get("/panel", response_class=HTMLResponse)
async def panel():
    """صفحهٔ پنل ادمین (از فایل admin_panel.html سرو می‌شود)."""
    try:
        with open(_PANEL_PATH, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h3>admin_panel.html یافت نشد</h3>", status_code=500)


# ===========================================================
# Entry Point
# ===========================================================

if __name__ == "__main__":
    # اطمینان از وجود مدل روی سرویس Ollama (در صورت نبود، یک‌بار دانلود می‌شود)
    ensure_ollama_model()

    # بارگذاری اسناد موجود از پوشهٔ آپلود و افزودن همهٔ آن‌ها به RAG
    load_existing_documents(language='fas')

    # هم‌گام‌سازی دیتابیس مدیریت فایل‌ها با وضعیت فعلی ایندکس (best-effort)
    _safe_db(sync_documents_db)

    if index is not None:
        stats = get_document_stats()
        print("✅ ایندکس آماده است:")
        print(f"   📄 {stats['total_documents']} سند")
        print(f"   📦 {stats['total_chunks']} chunk")
    else:
        print("⚠️  هنوز سندی آپلود نشده - ایندکس خالی است")
        print("📤 برای شروع، فایل‌های PDF یا تصویر خود را آپلود کنید")

    print(f"🔐 پنل ادمین: http://localhost:8000/panel  (نام کاربری: {ADMIN_USERNAME})")

    # (اختیاری) اجرای رابط گرافیکی Gradio در کنار سرور:
    # chat_ui.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    # اجرای سرور FastAPI در نخ اصلی (مسدودکننده تا فرایند زنده بماند)
    print("✅ FastAPI Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
