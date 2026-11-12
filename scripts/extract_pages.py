"""Extract every PDF page into corpus/pages.jsonl.

Usage:
    python scripts/extract_pages.py data/reports/midea-2024-combined.pdf
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF pages to JSONL")
    parser.add_argument("pdf", type=Path, help="path to the PDF")
    parser.add_argument("--out", type=Path, default=Path("corpus/pages.jsonl"), help="output JSONL")
    args = parser.parse_args()

    try:
        import pypdf
    except ImportError as exc:
        raise SystemExit("pypdf is required: python -m pip install pypdf") from exc

    reader = pypdf.PdfReader(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    empty = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                empty += 1
            handle.write(json.dumps({"page": number, "text": text}, ensure_ascii=False) + "\n")
    print(f"pages={len(reader.pages)} empty={empty} out={args.out}")


if __name__ == "__main__":
    main()
