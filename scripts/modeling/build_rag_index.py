"""Build the writable application RAG index with optional PDF references."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model" / "rag_index" / "export"
EXPORT = ROOT / "rag" / "index" / "export"


def chunks(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def pdf_rows(path: Path, size: int, overlap: int) -> list[dict[str, Any]]:
    doc_id = f"pdf_{hashlib.sha1(path.name.encode('utf-8')).hexdigest()[:12]}"
    coupon = "쿠폰" in path.stem
    axis = "discount_coupon" if coupon else "general"
    extra = None if coupon else ["discount_coupon", "set_bundle", "delivery", "store_menu_location"]
    rows = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        for part_no, text in enumerate(chunks(page.extract_text() or "", size, overlap)):
            rows.append({"chunk_id": f"{doc_id}_{page_no:03d}_{part_no:02d}", "doc_id": doc_id,
                "tier": "academic", "axis": axis, "axis_extra": extra,
                "claim_type": "academic_reference", "source_url": path.name,
                "page_start": page_no, "page_end": page_no, "contains_stat": False,
                "stat_values": [], "text": text})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not (EXPORT / "manifest.json").exists():
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE, EXPORT)
    manifest = json.loads((EXPORT / "manifest.json").read_text(encoding="utf-8"))
    old = [json.loads(line) for line in (EXPORT / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = np.load(EXPORT / "embeddings.npy").astype("float32")
    known = {row["doc_id"] for row in old}
    additions = [row for path in args.pdfs for row in pdf_rows(path, manifest.get("chunk_chars", 900), manifest.get("overlap", 150)) if row["doc_id"] not in known]
    if not additions:
        print("No new PDF chunks.")
        return
    model = SentenceTransformer(args.model, device="cpu", local_files_only=True)
    added_vectors = model.encode([row["text"] for row in additions], normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
    all_rows = old + additions
    all_vectors = np.vstack([vectors, added_vectors])
    tier_counts, axis_counts = {}, {}
    for row in all_rows:
        tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
        axis_counts[row["axis"]] = axis_counts.get(row["axis"], 0) + 1
    manifest.update({"n_chunks": len(all_rows), "dim": int(all_vectors.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(), "tier_counts": tier_counts,
        "axis_counts": axis_counts, "supplemental_pdfs": [path.name for path in args.pdfs]})
    np.save(EXPORT / "embeddings.npy", all_vectors)
    (EXPORT / "chunks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    (EXPORT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Added {len(additions)} PDF chunks; total={len(all_rows)}")


if __name__ == "__main__":
    main()
