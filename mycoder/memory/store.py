"""Session store — save and load conversation sessions as JSON."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    def __init__(self, base_dir: Path | str | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".mycoder" / "sessions"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def new_session(self, model: str, cwd: str) -> dict:
        return {
            "id": uuid.uuid4().hex[:8],
            "created": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "cwd": cwd,
            "messages": [],
        }

    def save(self, session: dict) -> Path:
        path = self.base_dir / f"{session['id']}.json"
        path.write_text(json.dumps(session, indent=2, default=str))
        path.chmod(0o600)
        return path

    def load(self, session_id: str) -> dict | None:
        path = self.base_dir / f"{session_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_latest(self) -> dict | None:
        files = sorted(self.base_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            return None
        return json.loads(files[-1].read_text())
