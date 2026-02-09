"""
Cover Letter Tab
Editable template with PDF preview for export.
"""

import base64
import streamlit as st
from weasyprint import HTML
import streamlit.components.v1 as components


DEFAULT_TEMPLATE = """[DATE]

USCIS

Re: O/P Visa Petition - [BENEFICIARY_NAME]

Dear Officer,

[INSERT COVER LETTER BODY HERE]

Sincerely,
[SIGNATURE]
"""


def render_cover_letter_tab():
    st.header("📝 Cover Letter")
    st.markdown("Edit the cover letter below. The PDF preview updates when you generate it.")

    if not st.session_state.cover_letter_template:
        st.session_state.cover_letter_template = DEFAULT_TEMPLATE

    if not st.session_state.cover_letter_text:
        st.session_state.cover_letter_text = st.session_state.cover_letter_template

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Editor")
        text = st.text_area(
            "Cover letter text",
            value=st.session_state.cover_letter_text,
            height=400
        )
        st.session_state.cover_letter_text = text

        if st.button("Generate PDF Preview", use_container_width=True):
            st.session_state.cover_letter_pdf = _text_to_pdf_bytes(text)

        if st.session_state.get("cover_letter_pdf"):
            st.download_button(
                "Download Cover Letter PDF",
                data=st.session_state.cover_letter_pdf,
                file_name="cover_letter.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with col2:
        st.subheader("Preview")
        if st.session_state.get("cover_letter_pdf"):
            _render_pdf_preview(st.session_state.cover_letter_pdf)
        else:
            st.info("Generate the PDF preview to view it here.")

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_cover_letter", use_container_width=True):
            st.session_state["goto_tab"] = "forms"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_cover_letter", use_container_width=True):
            st.session_state["goto_tab"] = "credentials"
            st.rerun()


def _text_to_pdf_bytes(text: str) -> bytes:
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


def _render_pdf_preview(pdf_bytes: bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
      src="data:application/pdf;base64,{b64}"
      width="100%"
      height="600"
      style="border: none;"
    ></iframe>
    """
    components.html(pdf_display, height=600)

