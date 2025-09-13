# app.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# import your pipeline helper
from stemi_pipeline import classify_path

app = FastAPI(title="STEMI Screening Demo")

# CORS (optional: allow your professor to call from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def upload_form(request: Request):
    """
    Simple browser form. Upload an ECG image and see the JSON result.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    JSON API. Receives a file (multipart/form-data; field name 'file'),
    returns the pipeline's JSON.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # enforce some basic file-type checks (optional)
    content_type = (file.content_type or "").lower()
    if not any(x in content_type for x in ["image/", "jpeg", "jpg", "png"]):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

    # Save to a secure temp file
    suffix = os.path.splitext(file.filename or "upload")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = classify_path(tmp_path)
        return JSONResponse(result)
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass