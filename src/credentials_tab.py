"""
Credentials Tab
Gather sources confirming credentials, awards, publications, and organizations.
"""

import streamlit as st


def render_credentials_tab():
    st.header("🏅 Credentials")
    st.markdown("Collect sources confirming awards, publications, and organization credentials.")

    results = st.session_state.credentials_results
    approvals = st.session_state.credentials_approvals

    st.divider()
    st.subheader("📥 Gather Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📤 Upload PDFs**")
        uploaded = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="credentials_upload",
            label_visibility="collapsed"
        )
        if uploaded:
            for file in uploaded:
                file_url = f"credentials_upload://{file.name}"
                if file_url not in results:
                    results[file_url] = {
                        "url": file_url,
                        "title": file.name,
                        "source": "Uploaded PDF",
                        "excerpt": f"User uploaded: {file.name}",
                        "pdf_bytes": file.read()
                    }
                    approvals[file_url] = True
            st.success(f"✅ {len(uploaded)} file(s)")

    with col2:
        st.markdown("**🔗 Paste URLs**")
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com/credentials",
            height=100,
            key="credentials_urls",
            label_visibility="collapsed"
        )
        if st.button("Add URLs", key="add_credentials_urls", use_container_width=True):
            urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
            for url in urls:
                if url not in results:
                    results[url] = {
                        "url": url,
                        "title": url.split("/")[-1] or "Credential Source",
                        "source": "URL",
                        "excerpt": f"Source: {url}"
                    }
                    approvals[url] = True
            st.success(f"✅ Added {len(urls)} URL(s)")
            st.rerun()

    st.divider()
    st.subheader("✅ Review & Approve")

    if not results:
        st.info("Add credential sources above.")
        return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Approve All", key="credentials_approve_all"):
            for key in results.keys():
                approvals[key] = True
            st.rerun()

    with col2:
        if st.button("❌ Reject All", key="credentials_reject_all"):
            for key in results.keys():
                approvals[key] = False
            st.rerun()

    with col3:
        if st.button("🗑️ Clear All", key="credentials_clear_all"):
            st.session_state.credentials_results = {}
            st.session_state.credentials_approvals = {}
            st.rerun()

    st.markdown("---")
    for i, item in enumerate(results.values()):
        approved = approvals.get(item["url"], True)
        st.checkbox(
            f"{item['title']} ({item['source']})",
            value=approved,
            key=f"credentials_approve_{i}"
        )
        approvals[item["url"]] = st.session_state[f"credentials_approve_{i}"]

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_credentials", use_container_width=True):
            st.session_state["goto_tab"] = "cover_letter"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_credentials", use_container_width=True):
            st.session_state["goto_tab"] = "final_package"
            st.rerun()

