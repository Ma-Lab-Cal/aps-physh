from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

_TAG_RE = re.compile(r"<[^>]+>")


def load_env_file(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from a dotenv-style file without adding a dependency."""
    import os

    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", str(text or ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def _field_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("text") or ""
    return normalize_text(str(value or ""))


def paper_text(record: dict[str, Any]) -> str:
    """Return title + abstract text from simple or APS-style metadata."""
    title = _field_text(record.get("title"))
    abstract = _field_text(record.get("abstract"))
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract or normalize_text(str(record.get("text_input") or record.get("text") or ""))


def paper_doi(record: dict[str, Any]) -> str:
    doi = record.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip()
    identifiers = record.get("identifiers")
    if isinstance(identifiers, dict):
        doi = identifiers.get("doi")
        if isinstance(doi, str) and doi.strip():
            return doi.strip()
    return ""


def paper_year(record: dict[str, Any]) -> str:
    for key in ("year", "date", "published", "pub_date", "publicationDate"):
        value = record.get(key)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            match = re.search(r"(18|19|20)\d{2}", value)
            if match:
                return match.group(0)
    return ""


def paper_id(record: dict[str, Any], text: str | None = None) -> str:
    doi = paper_doi(record)
    if doi:
        return f"doi:{doi}"
    if "paper_id" in record:
        return str(record["paper_id"])
    if "id" in record:
        return str(record["id"])
    digest = hashlib.sha1((text or paper_text(record)).encode("utf-8")).hexdigest()
    return f"text_sha1:{digest}"
