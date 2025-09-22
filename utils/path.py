# nautobot_mcp/utils/path.py

from pathlib import Path


def get_chroma_db_path() -> str:
    """
    Ensures and returns the absolute path to the Chroma DB directory under backend/nautobot_mcp/db.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == "nautobot_mcp":
            backend_path = parent / "backend" / "nautobot_mcp" / "db"
            backend_path.mkdir(parents=True, exist_ok=True)
            return str(backend_path)

    raise RuntimeError("Could not locate 'nautobot_mcp' folder in path hierarchy.")
