from pathlib import Path
import pandas as pd

def load_dataframe(path: str, nrows: int | None = 200) -> pd.DataFrame:
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