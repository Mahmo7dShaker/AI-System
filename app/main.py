from pathlib import Path
import uuid
import shutil
import json
import logging

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.deps import get_db
from app.db import engine, Base
from app.models import Dataset, DatasetMapping, ProcessedRun # إضافة ProcessedRun
from app.dataset_io import load_dataframe
from app.mapping import suggest_mapping
from ml.features.preprocess import preprocess_with_mapping
from ml.training.train_baselines import train_pipeline

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "datasets" / "01_raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "02_processed"
FEATURES_DIR = BASE_DIR / "datasets" / "03_features"
META_DIR = BASE_DIR / "datasets" / "metadata"
MODELS_DIR = BASE_DIR / "runtime" / "models"

for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, META_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static" 

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "solarmind_system.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE), 
        logging.StreamHandler()        
    ]
)

# -------- App --------
app = FastAPI(title="SolarMind")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# -------- Routes --------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
def upload_dataset(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "file"
    ext = filename.lower().split(".")[-1]
    if ext not in ("csv", "xlsx", "xls"):
        return {"ok": False, "error": "Only CSV/XLSX/XLS allowed"}
    
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    stored_path = RAW_DIR / stored_name # save in 01_raw

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    row = Dataset(original_name=filename, stored_name=stored_name, stored_path=str(stored_path))
    db.add(row)
    db.commit()
    db.refresh(row)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/dataset/{row.id}", status_code=303)
    return {"ok": True, "dataset_id": row.id}

@app.get("/dataset/{dataset_id}", response_class=HTMLResponse)
def dataset_page(dataset_id: int, request: Request, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds: return HTMLResponse("Not found", status_code=404)
    df = load_dataframe(ds.stored_path, nrows=50)
    suggestion = suggest_mapping(list(df.columns))
    return templates.TemplateResponse("dataset.html", {"request": request, "dataset": ds, "columns": list(df.columns), "suggestion": suggestion})

@app.post("/dataset/{dataset_id}/mapping")
def save_mapping(dataset_id: int, request: Request, datetime_col: str = Form(...), target_col: str = Form(...), db: Session = Depends(get_db), **kwargs):
    # save the mapping in the database for later use during processing
    mapping = {"datetime": datetime_col, "target": target_col, "weather": kwargs}
    row = DatasetMapping(dataset_id=dataset_id, mapping_json=json.dumps(mapping))
    db.add(row)
    db.commit()
    return RedirectResponse(url=f"/dataset/{dataset_id}", status_code=303)

@app.post("/dataset/{dataset_id}/process")
def process_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    last_map = db.query(DatasetMapping).filter(DatasetMapping.dataset_id == dataset_id).order_by(DatasetMapping.id.desc()).first()
    if not ds or not last_map: return {"ok": False, "error": "Missing data or mapping"}

    mapping = json.loads(last_map.mapping_json)
    df = load_dataframe(ds.stored_path)
    rows_before = len(df) # حساب عدد الصفوف قبل المعالجة
    
    processed_df, report = preprocess_with_mapping(df, mapping)
    rows_after = len(processed_df) # حساب عدد الصفوف بعد المعالجة

    # save the processed features in 03_features for later use in training
    out_path = FEATURES_DIR / f"ds_{dataset_id}_final.csv"
    processed_df.to_csv(out_path, index=False)

    # save a record of this processing run in the database
    new_run = ProcessedRun(
        dataset_id=dataset_id,
        mapping_id=last_map.id,
        processed_path=str(out_path),
        rows_before=rows_before,
        rows_after=rows_after
    )
    db.add(new_run)
    db.commit()

    return {"ok": True, "path": str(out_path), "report": report}

@app.post("/dataset/{dataset_id}/train")
def train_model(dataset_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    
    feature_path = FEATURES_DIR / f"ds_{dataset_id}_final.csv"
    if not feature_path.exists():
        return {"ok": False, "error": "Processed file not found. Run preprocessing first."}
    
    background_tasks.add_task(train_pipeline, feature_path)
    return {"ok": True, "message": "Training started in background"}