"""
نسخهٔ سبک سرور فقط برای فاز تست — فقط Endpoint /ask فعال است.

این فایل عمداً هیچ‌کدام از کتابخانه‌های سنگین ML (torch, faiss, transformers,
sentence-transformers, gradio, pytesseract, ...) را ایمپورت نمی‌کند تا ایمیج داکر
کوچک بماند و سریع بالا بیاید. منطق /ask دقیقاً همان نسخهٔ اصلی است
(llm_script_final.py) که فعلاً همیشه «سلام» برمی‌گرداند.

اتصال به دیتابیس اختیاری است: اگر DB در دسترس نباشد، پاسخ همراه با warning
برگردانده می‌شود و سرور بالا می‌ماند.
"""

import os
import time
import secrets
from datetime import datetime

from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

import mysql.connector


# ===========================================================
# تنظیمات اتصال به دیتابیس (قابل‌تنظیم با متغیّر محیطی)
# ===========================================================
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


# ===========================================================
# ورود ادمین (نام کاربری/رمز عبور پنل)
# ===========================================================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")
ADMIN_TOKEN_TTL = int(os.getenv("ADMIN_TOKEN_TTL", str(12 * 3600)))

_ADMIN_TOKENS = {}  # token -> زمان انقضا (epoch seconds)


def _issue_admin_token() -> str:
    token = secrets.token_urlsafe(32)
    _ADMIN_TOKENS[token] = time.time() + ADMIN_TOKEN_TTL
    return token


def _token_valid(token: str) -> bool:
    exp = _ADMIN_TOKENS.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        _ADMIN_TOKENS.pop(token, None)
        return False
    return True


def require_admin(authorization: str = Header(None)):
    """وابستگی FastAPI: اعتبارسنجی توکن ادمین از هدر Authorization: Bearer <token>."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not _token_valid(token):
        raise HTTPException(status_code=401, detail="نیاز به ورود ادمین")
    return token


def init_llm_history_table() -> None:
    """ایجاد جدول تاریخچه در صورت عدم وجود (idempotent)."""
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
    """ذخیرهٔ یک گفتگو و برگرداندن (شماره سؤال، زمان ثبت)."""
    last_err = None
    for _attempt in range(3):
        conn = get_llm_db_connection()
        try:
            cur = conn.cursor()
            conn.start_transaction()

            cur.execute(
                "SELECT COALESCE(MAX(question_number), 0) + 1 "
                "FROM llm_chat_history WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            question_number = int(cur.fetchone()[0])

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
            conn.rollback()
            last_err = e
        finally:
            conn.close()
    raise last_err if last_err else RuntimeError("ذخیرهٔ تاریخچه ناموفق بود")


def fetch_chat_history(user_id: int = None, limit: int = 200):
    """
    خواندن تاریخچهٔ گفتگوها برای نمایش در پنل.
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


def generate_llm_answer(question: str) -> str:
    """تولید پاسخ ربات. فعلاً بدون توجه به محتوای سؤال همیشه «سلام» برمی‌گرداند."""
    return "سلام"


# ===========================================================
# FastAPI App
# ===========================================================
app = FastAPI(title="LLM Bot (ask-only / test)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    try:
        init_llm_history_table()
        print("✅ جدول llm_chat_history آماده است")
    except Exception as _e:
        print(f"⚠️  اتصال/ساخت جدول تاریخچه ممکن نشد (در زمان درخواست دوباره تلاش می‌شود): {_e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask_endpoint(request: Request):
    """
    Endpoint اصلی ربات LLM.
    ورودی (JSON): {"user_id": <int>, "question": "<str>"}
    خروجی (JSON): {user_id, question_number, question, answer, created_at}
    """
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

    answer = generate_llm_answer(question)

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

    return {
        "user_id": user_id,
        "question_number": question_number,
        "question": question,
        "answer": answer,
        "created_at": created_at,
    }


# ===========================================================
# پنل مشاهدهٔ پیام‌ها
# ===========================================================

@app.post("/admin/login")
async def admin_login(request: Request):
    """ورود ادمین. ورودی JSON: {username, password} → خروجی: {token, expires_in}."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "بدنهٔ درخواست JSON معتبر نیست"})

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not (secrets.compare_digest(username, ADMIN_USERNAME)
            and secrets.compare_digest(password, ADMIN_PASSWORD)):
        return JSONResponse(status_code=401, content={"error": "نام کاربری یا رمز عبور نادرست است"})

    return {"token": _issue_admin_token(), "expires_in": ADMIN_TOKEN_TTL}


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
        # اگر دیتابیس در دسترس نباشد، خطای قابل‌فهم برگردان (پنل آن را نمایش می‌دهد)
        return JSONResponse(status_code=503, content={"error": f"اتصال به دیتابیس ممکن نشد: {e}"})


# مسیر فایل HTML پنل ادمین (کنار همین اسکریپت قرار دارد).
_PANEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin_panel.html")


@app.get("/panel", response_class=HTMLResponse)
async def panel():
    """صفحهٔ پنل ادمین مشاهدهٔ پیام‌ها (از فایل admin_panel.html سرو می‌شود)."""
    try:
        with open(_PANEL_PATH, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h3>admin_panel.html یافت نشد</h3>", status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
