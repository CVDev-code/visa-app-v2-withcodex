"""
Beneficiary Documents Tab
Upload passports and other documents per beneficiary.
"""

import streamlit as st


def render_beneficiary_docs_tab():
    st.header("📁 Beneficiary Documents")
    st.markdown("Upload passports and other beneficiary-specific documents.")

    beneficiaries = st.session_state.get("beneficiaries", [])
    if not beneficiaries:
        st.info("Add beneficiary details first in the Beneficiary Details tab.")
        return

    groups = st.session_state.get("beneficiary_groups", [])
    if not groups:
        groups = [beneficiaries]

    st.caption("P visas are grouped into packets of up to 25 beneficiaries.")

    for group_idx, group in enumerate(groups, start=1):
        with st.expander(f"Packet Group {group_idx} ({len(group)} beneficiaries)", expanded=False):
            for b in group:
                _render_beneficiary_uploads(b)

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_beneficiary_docs", use_container_width=True):
            st.session_state["goto_tab"] = "highlight"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_beneficiary_docs", use_container_width=True):
            st.session_state["goto_tab"] = "beneficiary_details"
            st.rerun()


def _render_beneficiary_uploads(beneficiary):
    name = beneficiary.get("name", "Unknown")
    st.subheader(name)

    if name not in st.session_state.beneficiary_docs:
        st.session_state.beneficiary_docs[name] = {
            "passports": [],
            "other_docs": []
        }

    passport_files = st.file_uploader(
        f"Passports for {name}",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"passports_{name}"
    )

    other_files = st.file_uploader(
        f"Other documents for {name}",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"other_docs_{name}"
    )

    if passport_files:
        st.session_state.beneficiary_docs[name]["passports"] = [
            {"filename": f.name, "bytes": f.read()} for f in passport_files
        ]

    if other_files:
        st.session_state.beneficiary_docs[name]["other_docs"] = [
            {"filename": f.name, "bytes": f.read()} for f in other_files
        ]

    st.caption(
        f"Saved: {len(st.session_state.beneficiary_docs[name]['passports'])} passport(s), "
        f"{len(st.session_state.beneficiary_docs[name]['other_docs'])} other doc(s)"
    )

