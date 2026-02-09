"""
Packet builder utilities for final package assembly.
"""

import io
from typing import List, Dict
from pypdf import PdfReader, PdfWriter
from weasyprint import HTML

from src.index_builder import build_index_pdf


def chunk_beneficiaries(beneficiaries, max_size=25):
    return [beneficiaries[i:i + max_size] for i in range(0, len(beneficiaries), max_size)]


def pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return 1


def merge_pdfs(pdf_list: List[bytes]) -> bytes:
    writer = PdfWriter()
    for pdf_bytes in pdf_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read()


def text_to_pdf_bytes(text: str) -> bytes:
    safe_text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f"""
    <html>
      <body>
        <pre style="font-family: Arial; font-size: 12px; white-space: pre-wrap;">
{safe_text}
        </pre>
      </body>
    </html>
    """
    return HTML(string=html).write_pdf()


def build_packet(packet_title: str, documents: List[Dict]) -> Dict:
    """
    documents: list of dicts with
      - label: str
      - pdf_bytes: bytes
    Returns dict with merged_pdf and index_pdf and entries.
    """
    entries = []
    current_page = 1
    for doc in documents:
        entries.append({"label": doc["label"], "page": current_page})
        pages = pdf_page_count(doc["pdf_bytes"])
        current_page += pages

    index_pdf = build_index_pdf(entries, packet_title=packet_title)
    merged_pdf = merge_pdfs([index_pdf] + [d["pdf_bytes"] for d in documents])

    return {
        "index_entries": entries,
        "index_pdf": index_pdf,
        "merged_pdf": merged_pdf
    }

