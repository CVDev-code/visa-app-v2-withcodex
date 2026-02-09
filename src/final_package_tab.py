"""
Final Package Tab
Compile all documents into per-packet PDFs with an index.
"""

import base64
import io
import zipfile
from typing import List

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.forms_tab import FORMS
from src.prompts import CRITERIA
from src.index_builder import schedule_for_criterion
from src.metadata import autodetect_metadata
from src.pdf_text import extract_text_from_pdf_bytes
from src.pdf_highlighter import annotate_pdf_bytes
from src.packet_builder import build_packet, text_to_pdf_bytes


def render_final_package_tab():
    st.header("📦 Final Package")
    st.markdown("Compile documents into packets and export a single PDF per packet.")

    beneficiaries = st.session_state.get("beneficiaries", [])
    if not beneficiaries:
        st.info("Add beneficiary details before generating packets.")
        return

    groups = st.session_state.get("beneficiary_groups", [])
    if not groups:
        groups = [beneficiaries]

    st.caption(f"Packets to generate: {len(groups)}")

    if st.button("Generate Packets", type="primary", use_container_width=True):
        _generate_packets(groups)

    packets = st.session_state.get("final_packets", {})
    if not packets:
        st.info("Generate packets to preview and download.")
        return

    packet_names = list(packets.keys())
    selected = st.selectbox("Preview Packet", packet_names)
    packet = packets[selected]

    st.download_button(
        "Download Selected Packet PDF",
        data=packet["merged_pdf"],
        file_name=f"{selected}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    _render_pdf_preview(packet["merged_pdf"])

    st.divider()
    if st.button("Prepare ZIP of All Packets", use_container_width=True):
        st.session_state.final_packets_zip = _build_packets_zip(packets)

    if st.session_state.get("final_packets_zip"):
        st.download_button(
            "Download ZIP",
            data=st.session_state.final_packets_zip,
            file_name="visa_packets.zip",
            mime="application/zip",
            use_container_width=True
        )

    st.divider()
    if st.button("← Back", key="nav_back_final_package", use_container_width=True):
        st.session_state["goto_tab"] = "credentials"
        st.rerun()


def _generate_packets(groups: List[List[dict]]):
    packets = {}
    for idx, group in enumerate(groups, start=1):
        group_names = ", ".join([b["name"] for b in group if b.get("name")])
        packet_name = f"packet_{idx}"
        packet_title = f"Index - Packet {idx} ({group_names})"

        documents = []
        documents.extend(_build_forms_documents(group))
        documents.append(_build_cover_letter_document())
        documents.extend(_build_beneficiary_docs(group))
        documents.extend(_build_evidence_documents())
        documents.extend(_build_credentials_documents())

        packet_result = build_packet(packet_title, documents)
        packets[packet_name] = packet_result

    st.session_state.final_packets = packets
    st.success("Packets generated.")


def _build_forms_documents(group: List[dict]):
    documents = []
    forms_state = st.session_state.forms_state
    standard_files = forms_state.get("standard_files", {})

    for form in FORMS:
        form_id = form["id"]
        form_label = form["label"]

        if form_id in standard_files:
            pdf_bytes = standard_files[form_id]["bytes"]
        else:
            standard_text = forms_state["standard"].get(form_id, "")
            overrides = []
            for b in group:
                name = b.get("name")
                override = forms_state["per_beneficiary"].get(name, {}).get(form_id, "")
                if override:
                    overrides.append(f"{name} override:\n{override}")

            combined = standard_text
            if overrides:
                combined += "\n\nOverrides:\n" + "\n\n".join(overrides)
            if not combined.strip():
                combined = f"{form_label} placeholder content."

            pdf_bytes = text_to_pdf_bytes(combined)

        documents.append({
            "label": f"Schedule 1: {form_label}",
            "pdf_bytes": pdf_bytes
        })

    return documents


def _build_cover_letter_document():
    text = st.session_state.get("cover_letter_text", "").strip()
    if not text:
        text = "Cover letter placeholder."
    pdf_bytes = text_to_pdf_bytes(text)
    return {
        "label": "Schedule 1: Cover Letter",
        "pdf_bytes": pdf_bytes
    }


def _build_beneficiary_docs(group: List[dict]):
    documents = []
    docs = st.session_state.get("beneficiary_docs", {})

    for b in group:
        name = b.get("name", "Unknown")
        b_docs = docs.get(name, {})

        for item in b_docs.get("passports", []):
            pdf_bytes = _ensure_pdf_bytes(item["filename"], item["bytes"])
            documents.append({
                "label": f"Schedule 2: {name} - Passport",
                "pdf_bytes": pdf_bytes
            })

        for item in b_docs.get("other_docs", []):
            pdf_bytes = _ensure_pdf_bytes(item["filename"], item["bytes"])
            documents.append({
                "label": f"Schedule 2: {name} - {item['filename']}",
                "pdf_bytes": pdf_bytes
            })

    return documents


def _build_evidence_documents():
    documents = []
    highlight_results = st.session_state.get("highlight_results", {})
    criterion_pdfs = st.session_state.get("criterion_pdfs", {})
    highlight_approvals = st.session_state.get("highlight_approvals", {})
    annotated_pdfs = st.session_state.get("annotated_pdfs", {})

    for cid, desc in CRITERIA.items():
        schedule = schedule_for_criterion(cid)
        if cid in highlight_results:
            for filename, data in highlight_results[cid].items():
                pdf_bytes = data.get("pdf_bytes")
                if not pdf_bytes:
                    continue

                quotes_dict = data.get("quotes", {})
                skip_highlight = data.get("skip_highlighting", False)
                approved_quotes = []

                if not skip_highlight:
                    file_approvals = highlight_approvals.get(cid, {}).get(filename, {})
                    for criterion_id, quote_list in quotes_dict.items():
                        for quote_data in quote_list:
                            quote_text = quote_data.get("quote", "")
                            quote_key = quote_text[:100]
                            if file_approvals.get(quote_key, True):
                                approved_quotes.append(quote_text)

                try:
                    if skip_highlight:
                        annotated_pdf = pdf_bytes
                    elif annotated_pdfs.get(cid, {}).get(filename):
                        annotated_pdf = annotated_pdfs[cid][filename]
                    else:
                        pdf_text = extract_text_from_pdf_bytes(pdf_bytes)
                        detected_meta = autodetect_metadata(pdf_text)
                        meta = {
                            "source_url": detected_meta.get("source_url", ""),
                            "venue_name": detected_meta.get("venue_name", ""),
                            "ensemble_name": detected_meta.get("ensemble_name", ""),
                            "performance_date": detected_meta.get("performance_date", ""),
                            "beneficiary_name": st.session_state.beneficiary_name,
                            "beneficiary_variants": st.session_state.beneficiary_variants,
                        }
                        annotated_pdf, _stats = annotate_pdf_bytes(
                            pdf_bytes=pdf_bytes,
                            quote_terms=approved_quotes,
                            criterion_id=cid,
                            meta=meta
                        )
                        if cid not in st.session_state.annotated_pdfs:
                            st.session_state.annotated_pdfs[cid] = {}
                        st.session_state.annotated_pdfs[cid][filename] = annotated_pdf
                except Exception:
                    annotated_pdf = pdf_bytes

                documents.append({
                    "label": f"{schedule}: {desc} - {filename}",
                    "pdf_bytes": annotated_pdf
                })
        elif cid in criterion_pdfs:
            for filename, pdf_bytes in criterion_pdfs[cid].items():
                documents.append({
                    "label": f"{schedule}: {desc} - {filename}",
                    "pdf_bytes": pdf_bytes
                })

    return documents


def _build_credentials_documents():
    documents = []
    results = st.session_state.get("credentials_results", {})
    approvals = st.session_state.get("credentials_approvals", {})

    for item in results.values():
        if not approvals.get(item.get("url"), True):
            continue
        pdf_bytes = item.get("pdf_bytes")
        if not pdf_bytes:
            continue
        documents.append({
            "label": f"Schedule 5: Credentials - {item.get('title', 'Credential')}",
            "pdf_bytes": pdf_bytes
        })
    return documents


def _ensure_pdf_bytes(filename: str, data: bytes) -> bytes:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return data
    if lower.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        output = io.BytesIO()
        img.save(output, format="PDF")
        output.seek(0)
        return output.read()
    return data


def _build_packets_zip(packets: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, packet in packets.items():
            zip_file.writestr(f"{name}.pdf", packet["merged_pdf"])
    buffer.seek(0)
    return buffer.read()


def _render_pdf_preview(pdf_bytes: bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
      src="data:application/pdf;base64,{b64}"
      width="100%"
      height="700"
      style="border: none;"
    ></iframe>
    """
    components.html(pdf_display, height=700)

