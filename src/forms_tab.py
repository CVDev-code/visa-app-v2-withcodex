"""
Forms Tab
Placeholder workflow for five forms with standard fields and overrides.
"""

import streamlit as st


FORMS = [
    {"id": "form_1", "label": "Form 1 (Placeholder)"},
    {"id": "form_2", "label": "Form 2 (Placeholder)"},
    {"id": "form_3", "label": "Form 3 (Placeholder)"},
    {"id": "form_4", "label": "Form 4 (Placeholder)"},
    {"id": "form_5", "label": "Form 5 (Placeholder)"}
]


def render_forms_tab():
    st.header("🧩 Forms")
    st.markdown("Set standard information once, then override per beneficiary as needed.")

    if "standard_files" not in st.session_state.forms_state:
        st.session_state.forms_state["standard_files"] = {}

    _render_standard_forms()
    st.divider()
    _render_per_beneficiary_overrides()

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_forms", use_container_width=True):
            st.session_state["goto_tab"] = "beneficiary_details"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_forms", use_container_width=True):
            st.session_state["goto_tab"] = "cover_letter"
            st.rerun()


def _render_standard_forms():
    st.subheader("Standard Information (applies to all packets)")

    for form in FORMS:
        form_id = form["id"]
        form_label = form["label"]

        with st.expander(form_label, expanded=False):
            standard_text = st.text_area(
                "Standard data",
                value=st.session_state.forms_state["standard"].get(form_id, ""),
                placeholder="Enter standard data that usually repeats.",
                height=80,
                key=f"standard_text_{form_id}"
            )

            st.session_state.forms_state["standard"][form_id] = standard_text

            uploaded = st.file_uploader(
                "Upload a PDF for this form (optional)",
                type=["pdf"],
                key=f"form_pdf_{form_id}"
            )
            if uploaded:
                st.session_state.forms_state["standard_files"][form_id] = {
                    "filename": uploaded.name,
                    "bytes": uploaded.read()
                }


def _render_per_beneficiary_overrides():
    st.subheader("Per Beneficiary Overrides")

    beneficiaries = st.session_state.get("beneficiaries", [])
    if not beneficiaries:
        st.info("Add beneficiary details to enable overrides.")
        return

    beneficiary_names = [b["name"] for b in beneficiaries if b.get("name")]
    selected = st.selectbox("Choose beneficiary", beneficiary_names)

    if selected not in st.session_state.forms_state["per_beneficiary"]:
        st.session_state.forms_state["per_beneficiary"][selected] = {}

    for form in FORMS:
        form_id = form["id"]
        form_label = form["label"]

        override_text = st.text_area(
            f"{form_label} override",
            value=st.session_state.forms_state["per_beneficiary"][selected].get(form_id, ""),
            placeholder="Only enter data that differs for this beneficiary.",
            height=80,
            key=f"override_{selected}_{form_id}"
        )

        st.session_state.forms_state["per_beneficiary"][selected][form_id] = override_text

