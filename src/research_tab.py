"""
Research Tab - Tab 1
Gather evidence with 3 input methods per criterion in dropdown format
"""

import streamlit as st
from src.prompts import CRITERIA


def render_research_tab():
    """
    Main research interface with dropdowns for each criterion
    """
    
    st.header("🔍 Research & Gather Evidence")
    st.markdown("Gather sources for each criterion using upload, URLs, or AI agent")
    
    beneficiary_name = st.session_state.beneficiary_name
    
    # Show all criteria as dropdowns
    st.divider()
    
    for cid, desc in CRITERIA.items():
        render_criterion_research(cid, desc, beneficiary_name)
    
    # Summary at bottom
    st.divider()
    render_research_summary()


def render_criterion_research(cid: str, desc: str, beneficiary_name: str):
    """
    Single criterion research section with 3 input methods + approval
    """
    
    # Count current results
    results = st.session_state.research_results.get(cid, [])
    approvals = st.session_state.research_approvals.get(cid, {})
    approved_count = sum(1 for url, approved in approvals.items() if approved)
    
    status = f"({len(results)} sources, {approved_count} approved)" if results else ""
    
    # Default to collapsed UNLESS there are results (to keep it open during interaction)
    # This prevents the annoying auto-close behavior
    is_expanded = len(results) > 0
    
    with st.expander(f"📋 **Criterion ({cid}):** {desc} {status}", expanded=is_expanded):
        
        # ========================================
        # 3 INPUT METHODS
        # ========================================
        st.markdown("### 📥 Gather Sources")
        
        col1, col2, col3 = st.columns(3)
        
        # METHOD 1: Upload
        with col1:
            st.markdown("**📤 Upload PDFs**")
            uploaded = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"upload_{cid}",
                label_visibility="collapsed"
            )
            
            if uploaded:
                # Convert uploads to "results" format
                if cid not in st.session_state.research_results:
                    st.session_state.research_results[cid] = []
                if cid not in st.session_state.research_approvals:
                    st.session_state.research_approvals[cid] = {}
                
                for file in uploaded:
                    file_url = f"upload://{file.name}"
                    
                    # Check if already added
                    if not any(r['url'] == file_url for r in st.session_state.research_results[cid]):
                        st.session_state.research_results[cid].append({
                            'url': file_url,
                            'title': file.name,
                            'source': 'Uploaded PDF',
                            'excerpt': f'User uploaded: {file.name}',
                            'pdf_bytes': file.read()
                        })
                        st.session_state.research_approvals[cid][file_url] = True
                
                st.success(f"✅ {len(uploaded)} file(s)")
        
        # METHOD 2: URLs
        with col2:
            st.markdown("**🔗 Paste URLs**")
            urls_text = st.text_area(
                "Enter URLs (one per line)",
                placeholder="https://example.com/article",
                height=100,
                key=f"urls_{cid}",
                label_visibility="collapsed"
            )
            
            if st.button("Add URLs", key=f"add_urls_{cid}", use_container_width=True):
                urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
                
                if urls:
                    if cid not in st.session_state.research_results:
                        st.session_state.research_results[cid] = []
                    if cid not in st.session_state.research_approvals:
                        st.session_state.research_approvals[cid] = {}
                    
                    for url in urls:
                        # Check if already added
                        if not any(r['url'] == url for r in st.session_state.research_results[cid]):
                            st.session_state.research_results[cid].append({
                                'url': url,
                                'title': url.split('/')[-1] or 'Article',
                                'source': 'URL',
                                'excerpt': f'Source: {url}'
                            })
                            st.session_state.research_approvals[cid][url] = True
                    
                    st.success(f"✅ Added {len(urls)} URL(s)")
                    st.rerun()
        
        # METHOD 3: AI Agent
        with col3:
            st.markdown("**🤖 AI Agent**")
            
            if st.button("🔍 Search with AI", key=f"ai_{cid}", use_container_width=True):
                with st.spinner("🤖 AI Agent searching..."):
                    try:
                        from src.ai_responses import search_with_responses_api
                        
                        results_found = search_with_responses_api(
                            artist_name=beneficiary_name,
                            criterion_id=cid,
                            criterion_description=desc,
                            name_variants=st.session_state.beneficiary_variants,
                            artist_field=st.session_state.artist_field,
                            max_results=10
                        )
                        
                        if results_found:
                            # Add to research results
                            if cid not in st.session_state.research_results:
                                st.session_state.research_results[cid] = []
                            if cid not in st.session_state.research_approvals:
                                st.session_state.research_approvals[cid] = {}
                            
                            for item in results_found:
                                url = item['url']
                                if not any(r['url'] == url for r in st.session_state.research_results[cid]):
                                    st.session_state.research_results[cid].append(item)
                                    st.session_state.research_approvals[cid][url] = True
                            
                            st.success(f"✅ AI Agent found {len(results_found)} sources!")
                            st.rerun()
                        else:
                            st.warning("AI Agent found no results")
                    
                    except Exception as e:
                        st.error(f"AI Agent error: {str(e)}")
                        st.info("💡 Try the Upload or URL methods instead")
        
        # ========================================
        # REVIEW & APPROVE RESULTS
        # ========================================
        if not results:
            st.info("👆 Use one of the methods above to gather sources")
            return
        
        st.divider()
        st.markdown("### ✅ Review & Approve Sources")
        
        # Bulk actions
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Approve All", key=f"approve_all_{cid}"):
                for i, item in enumerate(results):
                    st.session_state.research_approvals[cid][item['url']] = True
                    st.session_state[f"approve_{cid}_{i}"] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Reject All", key=f"reject_all_{cid}"):
                for i, item in enumerate(results):
                    st.session_state.research_approvals[cid][item['url']] = False
                    st.session_state[f"approve_{cid}_{i}"] = False
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Results", key=f"clear_{cid}"):
                st.session_state.research_results[cid] = []
                st.session_state.research_approvals[cid] = {}
                st.rerun()
        
        st.markdown("---")
        
        # Show each result with checkbox
        for i, item in enumerate(results):
            url = item['url']
            title = item.get('title', 'Untitled')
            source = item.get('source', 'Unknown')
            excerpt = item.get('excerpt', '')
            
            # Get filename for skip_highlighting tracking
            # MUST match the filename created in convert_approved_to_pdfs
            if url.startswith('upload://'):
                filename = url.replace('upload://', '')
            else:
                filename = title + '.pdf'  # Add .pdf extension to match convert logic
            
            # Read approval state directly from session state (not local variable)
            is_approved = st.session_state.research_approvals[cid].get(url, True)
            
            # Initialize skip_highlighting for this criterion if needed
            if cid not in st.session_state.skip_highlighting:
                st.session_state.skip_highlighting[cid] = {}
            skip_highlight = st.session_state.skip_highlighting[cid].get(filename, False)
            
            # Checkbox for approval
            col_approve, col_skip = st.columns([3, 1])
            
            with col_approve:
                new_approval = st.checkbox(
                    f"**[{source}]** {title}",
                    value=is_approved,
                    key=f"approve_{cid}_{i}"
                )
                st.session_state.research_approvals[cid][url] = new_approval
            
            with col_skip:
                if is_approved:  # Only show skip option if approved
                    new_skip = st.checkbox(
                        "Skip highlighting",
                        value=skip_highlight,
                        key=f"skip_{cid}_{i}",
                        help="Include in export as-is without highlighting"
                    )
                    st.session_state.skip_highlighting[cid][filename] = new_skip
            
            # Show excerpt and URL
            if excerpt:
                st.caption(f"📝 {excerpt[:200]}...")
            st.caption(f"🔗 {url}")
            
            st.markdown("---")
        
        # Show counts
        approved = sum(1 for ok in st.session_state.research_approvals[cid].values() if ok)
        rejected = sum(1 for ok in st.session_state.research_approvals[cid].values() if not ok)
        st.write(f"**✅ Approved: {approved}** | **❌ Rejected: {rejected}**")
        
        # Regenerate option
        st.divider()
        st.markdown("### 🔄 Not satisfied?")
        
        feedback_text = st.text_area(
            "Tell AI what to improve",
            placeholder="e.g., 'Need more from major publications'",
            key=f"feedback_{cid}",
            height=60
        )
        
        if st.button("🔄 Regenerate with AI", key=f"regen_{cid}"):
            with st.spinner("Regenerating..."):
                try:
                    from src.ai_responses import search_with_responses_api
                    
                    # Get approved/rejected URLs
                    approved_urls = [url for url, ok in st.session_state.research_approvals[cid].items() if ok]
                    rejected_urls = [url for url, ok in st.session_state.research_approvals[cid].items() if not ok]
                    
                    # Build feedback message
                    feedback_msg = feedback_text or ""
                    
                    if rejected_urls:
                        feedback_msg += f"\n\nAvoid sources like these (rejected):\n"
                        for url in rejected_urls[:3]:  # Show first 3 as examples
                            feedback_msg += f"- {url}\n"
                    
                    new_results = search_with_responses_api(
                        artist_name=beneficiary_name,
                        criterion_id=cid,
                        criterion_description=desc,
                        name_variants=st.session_state.beneficiary_variants,
                        artist_field=st.session_state.artist_field,
                        feedback=feedback_msg,
                        max_results=10
                    )
                    
                    if new_results:
                        # Keep approved, add new
                        kept = [r for r in results if st.session_state.research_approvals[cid].get(r['url'], False)]
                        kept_urls = {r['url'] for r in kept}
                        new_items = [r for r in new_results if r['url'] not in kept_urls]
                        
                        st.session_state.research_results[cid] = kept + new_items
                        
                        # Auto-approve new
                        for item in new_items:
                            st.session_state.research_approvals[cid][item['url']] = True
                        
                        st.success(f"✅ Found {len(new_items)} new sources!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_research_summary():
    """Show summary and convert to PDFs button"""
    
    st.subheader("📊 Summary")
    
    total_sources = sum(len(results) for results in st.session_state.research_results.values())
    total_approved = sum(
        sum(1 for ok in approvals.values() if ok)
        for approvals in st.session_state.research_approvals.values()
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Sources", total_sources)
    with col2:
        st.metric("Approved Sources", total_approved)
    
    if total_approved == 0:
        st.info("No sources approved yet. Approve sources above to continue.")
        return
    
    st.divider()
    st.markdown(f"### 🔄 Ready to process {total_approved} approved sources")
    
    if st.button("🔄 Convert to PDFs & Continue to Highlight Tab", type="primary", use_container_width=True):
        with st.spinner(f"Processing {total_approved} sources..."):
            convert_approved_to_pdfs()

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.button("← Back", key="nav_back_research", disabled=True, use_container_width=True)
    with nav_col2:
        if st.button("Next Page →", key="nav_next_research", use_container_width=True):
            st.session_state["goto_tab"] = "highlight"
            st.rerun()


def convert_approved_to_pdfs():
    """Convert all approved sources to PDFs"""
    
    from src.web_to_pdf import batch_convert_urls_to_pdfs
    
    # Separate uploads from URLs
    for cid, results in st.session_state.research_results.items():
        approvals = st.session_state.research_approvals.get(cid, {})
        skip_flags = st.session_state.skip_highlighting.get(cid, {})
        
        if cid not in st.session_state.criterion_pdfs:
            st.session_state.criterion_pdfs[cid] = {}
        
        # Track which PDFs should skip highlighting
        if cid not in st.session_state.highlight_results:
            st.session_state.highlight_results[cid] = {}
        
        # Process each approved result
        urls_to_convert = []
        
        for item in results:
            url = item['url']
            
            if not approvals.get(url, False):
                continue  # Skip rejected
            
            # Determine filename
            if url.startswith('upload://'):
                filename = url.replace('upload://', '')
            else:
                filename = item.get('title', 'source') + '.pdf'
            
            # Check if this should skip highlighting
            should_skip = skip_flags.get(filename, False)
            
            # Check if upload
            if url.startswith('upload://'):
                # Already have PDF bytes
                pdf_bytes = item['pdf_bytes']
                st.session_state.criterion_pdfs[cid][filename] = pdf_bytes
                
                # If skip highlighting, mark it in highlight_results (bypasses AI analysis)
                if should_skip:
                    st.session_state.highlight_results[cid][filename] = {
                        'quotes': {},
                        'notes': 'Document marked to skip highlighting - included as-is',
                        'pdf_bytes': pdf_bytes,
                        'skip_highlighting': True
                    }
            else:
                # URL to convert - ALL URLs need to be converted to PDF
                urls_to_convert.append({
                    'url': url,
                    'title': item.get('title', 'source'),
                    'filename': filename,
                    'skip_highlighting': should_skip  # Track skip flag for later
                })
        
        # Convert URLs to PDFs
        if urls_to_convert:
            try:
                pdfs = batch_convert_urls_to_pdfs(
                    {cid: urls_to_convert},
                    progress_callback=None
                )
                
                if cid in pdfs:
                    for filename, pdf_bytes in pdfs[cid].items():
                        # Store the PDF
                        st.session_state.criterion_pdfs[cid][filename] = pdf_bytes
                        
                        # Find if this should skip highlighting
                        url_item = next((u for u in urls_to_convert if u.get('filename') == filename), None)
                        if url_item and url_item.get('skip_highlighting', False):
                            # Mark to skip AI analysis and annotation
                            st.session_state.highlight_results[cid][filename] = {
                                'quotes': {},
                                'notes': 'Document marked to skip highlighting - included as-is',
                                'pdf_bytes': pdf_bytes,
                                'skip_highlighting': True
                            }
            
            except Exception as e:
                st.error(f"Error converting criterion {cid}: {str(e)}")
    
    total_pdfs = sum(len(pdfs) for pdfs in st.session_state.criterion_pdfs.values())
    skipped_count = sum(
        1 for cid_data in st.session_state.highlight_results.values()
        for doc_data in cid_data.values()
        if doc_data.get('skip_highlighting', False)
    )
    highlight_count = total_pdfs - skipped_count
    
    msg = f"✅ Processed {total_pdfs} PDFs!\n\n"
    if skipped_count > 0:
        msg += f"📄 {highlight_count} will be highlighted\n"
        msg += f"🔒 {skipped_count} marked to skip highlighting (will be included as-is)\n\n"
    
    msg += "**Go to the Highlight & Export tab** to continue →"
    
    st.success(msg)
