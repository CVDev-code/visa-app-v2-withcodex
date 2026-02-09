"""
Beneficiary Details Tab
Collect and store beneficiary data for O and P visas.
"""

import csv
import io
import streamlit as st


def _chunk_beneficiaries(beneficiaries, chunk_size=25):
    return [beneficiaries[i:i + chunk_size] for i in range(0, len(beneficiaries), chunk_size)]


def render_beneficiary_details_tab():
    st.header("🧾 Details of Beneficiary(s)")
    st.markdown("Capture beneficiary details for O and P visas.")

    visa_type = st.session_state.get("visa_type", "O")

    if visa_type == "O":
        _render_single_beneficiary()
    else:
        _render_multiple_beneficiaries()

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_beneficiary_details", use_container_width=True):
            st.session_state["goto_tab"] = "beneficiary_docs"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_beneficiary_details", use_container_width=True):
            st.session_state["goto_tab"] = "forms"
            st.rerun()


def _render_single_beneficiary():
    st.subheader("O Visa: Single Beneficiary")

    name = st.session_state.beneficiary_name.strip()
    if not name:
        st.info("Enter the beneficiary name at the top of the page to continue.")
        return

    details = st.session_state.beneficiary_details.get(name, {})

    col1, col2, col3 = st.columns(3)
    with col1:
        dob = st.text_input("Date of Birth", value=details.get("dob", ""), placeholder="YYYY-MM-DD")
    with col2:
        nationality = st.text_input("Nationality", value=details.get("nationality", ""))
    with col3:
        passport = st.text_input("Passport Number", value=details.get("passport_number", ""))

    st.session_state.beneficiary_details[name] = {
        "name": name,
        "dob": dob,
        "nationality": nationality,
        "passport_number": passport
    }

    st.session_state.beneficiaries = [{
        "name": name,
        "dob": dob,
        "nationality": nationality,
        "passport_number": passport
    }]

    st.session_state.beneficiary_groups = _chunk_beneficiaries(st.session_state.beneficiaries, 25)

    st.success("Beneficiary details saved.")


def _render_multiple_beneficiaries():
    st.subheader("P Visa: Multiple Beneficiaries")
    st.caption("Upload a CSV for now, or paste names manually. Max 25 per packet.")

    manual_names = st.text_area(
        "Manual entry (one name per line)",
        placeholder="Beneficiary 1\nBeneficiary 2\nBeneficiary 3",
        height=120
    )

    uploaded = st.file_uploader(
        "Upload CSV (columns: name, dob, nationality, passport_number)",
        type=["csv"]
    )

    beneficiaries = []

    if uploaded:
        try:
            content = uploaded.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                if not row.get("name"):
                    continue
                beneficiaries.append({
                    "name": row.get("name", "").strip(),
                    "dob": row.get("dob", "").strip(),
                    "nationality": row.get("nationality", "").strip(),
                    "passport_number": row.get("passport_number", "").strip()
                })
        except Exception as exc:
            st.error(f"CSV parse error: {exc}")

    if not beneficiaries and manual_names.strip():
        for line in manual_names.split("\n"):
            name = line.strip()
            if name:
                beneficiaries.append({
                    "name": name,
                    "dob": "",
                    "nationality": "",
                    "passport_number": ""
                })

    if beneficiaries:
        st.session_state.beneficiaries = beneficiaries
        for b in beneficiaries:
            st.session_state.beneficiary_details[b["name"]] = b

        st.session_state.beneficiary_groups = _chunk_beneficiaries(beneficiaries, 25)

        st.success(f"Loaded {len(beneficiaries)} beneficiary(ies).")
        st.info(f"Packets needed: {len(st.session_state.beneficiary_groups)}")
    else:
        st.info("Add beneficiary data to continue.")

