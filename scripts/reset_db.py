from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "runtime" / "app.db"

if DB_PATH.exists():
    DB_PATH.unlink()
    print("Deleted:", DB_PATH)
else:
    print("No DB found:", DB_PATH)

print("Done. Run the server again to recreate tables.")