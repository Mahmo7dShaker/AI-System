#------------ dataset_io.py required for loading datasets into DataFrames ------------- 
from pathlib import Path
import pandas as pd # علشان نحول الملف ل DataFrame

def load_dataframe(path: str, nrows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        for enc in ("utf-8", "utf-8-sig", "cp1256", "latin1"):
            try:
                return pd.read_csv(p, nrows=nrows, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(p, nrows=nrows)

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, nrows=nrows)

    raise ValueError("Unsupported file extension")