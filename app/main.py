from pathlib import Path # for handling file paths in a clean way across OSes
import uuid # for generating unique filenames
import shutil # for saving uploaded files علشان ميتحملش كله على الram
import json

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # يربط FastAPI بـ Jinja templates.
from sqlalchemy.orm import Session

from app.deps import get_db #يرجع Session لكل request ويقفلها.
from app.db import engine, Base
from app.models import Dataset, DatasetMapping
from app.dataset_io import load_dataframe # read csv/xlsx into pandas DataFrame
from app.mapping import suggest_mapping #read columns aoto
from ml.features.preprocess import preprocess_with_mapping #مسؤول عن تجهيز data للـ ML:


# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent.parent # app/../ looking project root(project file) علشان لو حبيت اشغل الكود من مكان تاني او لو نقلت على جاهز تاني  ميظبطش دايما يبني اي 
#path بنسبه لroot بتاع المشروع مش بالنسبة لملف 

UPLOAD_DIR = BASE_DIR / "datasets" / "raw"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = BASE_DIR / "runtime" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# -------- App --------
app = FastAPI(title="SolarMind")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# -------- Basic --------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------- Upload --------
@app.post("/upload")
def upload_dataset(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "file"
    ext = filename.lower().split(".")[-1]
    if ext not in ("csv", "xlsx", "xls"):
        return {"ok": False, "error": "Only CSV/XLSX/XLS files are allowed"}
    # Generate UUID-based filename 
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    stored_path = UPLOAD_DIR / stored_name

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    row = Dataset(
        original_name=filename,
        stored_name=stored_name,
        stored_path=str(stored_path),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url=f"/dataset/{row.id}", status_code=303)

    return {"ok": True, "dataset_id": row.id, "stored_name": stored_name}


# -------- List datasets --------
@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    items = db.query(Dataset).order_by(Dataset.id.desc()).limit(20).all()
    return [
        {
            "id": x.id,
            "original_name": x.original_name,
            "stored_name": x.stored_name,
            "stored_path": x.stored_path,
            "uploaded_at": x.uploaded_at.isoformat() if x.uploaded_at else None,
        }
        for x in items
    ]


# -------- Dataset page + mapping --------
@app.get("/dataset/{dataset_id}", response_class=HTMLResponse)
def dataset_page(dataset_id: int, request: Request, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        return HTMLResponse("Dataset not found", status_code=404)

    df = load_dataframe(ds.stored_path, nrows=50)
    cols = list(df.columns)

    suggestion = suggest_mapping(cols)

    last_map = (
        db.query(DatasetMapping)
        .filter(DatasetMapping.dataset_id == dataset_id)
        .order_by(DatasetMapping.id.desc())
        .first()
    )
    saved = json.loads(last_map.mapping_json) if last_map else None

    preview_rows = df.head(8).fillna("").to_dict(orient="records")

    return templates.TemplateResponse(
        "dataset.html",
        {
            "request": request,
            "dataset": ds,
            "columns": cols,
            "preview": preview_rows,
            "suggestion": suggestion,
            "saved": saved,
        },
    )


@app.post("/dataset/{dataset_id}/mapping")
def save_mapping(
    dataset_id: int,
    request: Request,
    datetime_col: str = Form(...),
    target_col: str = Form(...),
    plant_id_col: str = Form(""),
    irr_col: str = Form(""),
    amb_col: str = Form(""),
    mod_col: str = Form(""),
    wind_col: str = Form(""),
    cloud_col: str = Form(""),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        return {"ok": False, "error": "Dataset not found"}

    mapping = {
        "datetime": datetime_col or None,
        "plant_id": plant_id_col or None,
        "target": target_col or None,
        "weather": {
            "irradiance": irr_col or None,
            "ambient_temp": amb_col or None,
            "module_temp": mod_col or None,
            "wind_speed": wind_col or None,
            "cloud_cover": cloud_col or None,
        },
    }

    row = DatasetMapping(
        dataset_id=dataset_id,
        mapping_json=json.dumps(mapping, ensure_ascii=False),
    )
    db.add(row)
    db.commit()

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url=f"/dataset/{dataset_id}", status_code=303)

    return {"ok": True, "dataset_id": dataset_id, "mapping_id": row.id}


# -------- Processing--------
@app.post("/dataset/{dataset_id}/process")
def process_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        return {"ok": False, "error": "Dataset not found"}

    last_map = (
        db.query(DatasetMapping)
        .filter(DatasetMapping.dataset_id == dataset_id)
        .order_by(DatasetMapping.id.desc())
        .first()
    )
    if not last_map:
        return {"ok": False, "error": "No mapping saved for this dataset"}

    mapping = json.loads(last_map.mapping_json)

    # load full dataset
    df = load_dataframe(ds.stored_path, nrows=None)
    rows_before = len(df)
    #start processing(pipelines) and get cleaned DataFrame the report
    processed_df, report = preprocess_with_mapping(df, mapping)
    rows_after = len(processed_df)

    out_path = PROCESSED_DIR / f"dataset_{dataset_id}_processed.csv"
    processed_df.to_csv(out_path, index=False)

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "dropped_rows": report.get("dropped_rows", 0),
        "processed_path": str(out_path),
        "columns": list(processed_df.columns),
    }