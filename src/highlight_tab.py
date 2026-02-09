"""
Highlight Tab - Tab 2
Highlight PDFs, review quotes, and export with criterion subfolders
"""

import streamlit as st
import io
import zipfile
from datetime import datetime
from src.prompts import CRITERIA


def render_highlight_tab():
    """
    Main highlight interface with dropdowns for each criterion
    """
    
    st.header("✨ Highlight & Export")
    st.markdown("Review highlighted quotes and export organized evidence package")
    
    # Check if we have PDFs
    total_pdfs = sum(len(pdfs) for pdfs in st.session_state.criterion_pdfs.values())
    
    if total_pdfs == 0:
        st.info("""
        📂 **No PDFs yet!**
        
        Go to **Research & Gather Evidence** tab to:
        1. Gather sources (upload, URL, or AI)
        2. Approve sources
        3. Convert to PDFs
        """)
        return
    
    # Count PDFs marked to skip highlighting
    skip_count = sum(
        1 for cid_data in st.session_state.highlight_results.values()
        for doc_data in cid_data.values()
        if doc_data.get('skip_highlighting', False)
    )
    
    highlight_count = total_pdfs - skip_count
    
    if skip_count > 0:
        st.success(f"📄 **{total_pdfs} PDF(s) ready** ({highlight_count} to highlight, {skip_count} to skip)")
    else:
        st.success(f"📄 **{total_pdfs} PDF(s) ready** to highlight")
    
    # Highlight all button
    if st.button("✨ Highlight All Criteria", type="primary", use_container_width=True):
        highlight_all_criteria()
    
    st.divider()
    
    # Show each criterion as dropdown
    for cid in CRITERIA.keys():
        if cid not in st.session_state.criterion_pdfs:
            continue
        
        pdfs = st.session_state.criterion_pdfs[cid]
        if not pdfs:
            continue
        
        render_criterion_highlights(cid)
    
    # Export at bottom
    st.divider()
    render_export_section()


def render_criterion_highlights(cid: str):
    """
    Single criterion highlight section with approval/regenerate
    """
    
    desc = CRITERIA.get(cid, "")
    pdfs = st.session_state.criterion_pdfs.get(cid, {})
    highlights = st.session_state.highlight_results.get(cid, {})
    
    # Count stats
    highlighted_count = len(highlights)
    total_quotes = sum(
        sum(len(quotes) for quotes in h.get('quotes', {}).values())
        for h in highlights.values()
    )
    
    status = f"({len(pdfs)} PDFs, {highlighted_count} highlighted, {total_quotes} quotes)" if highlights else f"({len(pdfs)} PDFs)"
    
    # Default to collapsed UNLESS there are highlights (to keep it open during interaction)
    is_expanded = highlighted_count > 0
    
    with st.expander(f"📋 **Criterion ({cid}):** {desc} {status}", expanded=is_expanded):
        
        # Highlight button for this criterion
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(f"✨ Highlight All in Criterion ({cid})", key=f"highlight_{cid}", use_container_width=True):
                highlight_criterion(cid)
                st.rerun()
        
        with col2:
            if st.button(f"🗑️ Clear", key=f"clear_highlights_{cid}", use_container_width=True):
                if cid in st.session_state.highlight_results:
                    del st.session_state.highlight_results[cid]
                if cid in st.session_state.highlight_approvals:
                    del st.session_state.highlight_approvals[cid]
                st.rerun()
        
        if not highlights:
            st.info(f"Click 'Highlight All' to analyze {len(pdfs)} PDF(s)")
            return
        
        st.divider()
        st.markdown("### 📝 Review Quotes")
        
        # Bulk actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve All Quotes", key=f"approve_all_quotes_{cid}"):
                # Initialize if needed
                if cid not in st.session_state.highlight_approvals:
                    st.session_state.highlight_approvals[cid] = {}
                
                for filename, data in highlights.items():
                    if filename not in st.session_state.highlight_approvals[cid]:
                        st.session_state.highlight_approvals[cid][filename] = {}
                    for criterion_id, quotes in data.get('quotes', {}).items():
                        for i, quote in enumerate(quotes):
                            quote_key = quote['quote'][:100]  # Use first 100 chars as key
                            st.session_state.highlight_approvals[cid][filename][quote_key] = True
                            st.session_state[f"quote_{cid}_{filename}_{i}"] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Reject All Quotes", key=f"reject_all_quotes_{cid}"):
                # Initialize if needed
                if cid not in st.session_state.highlight_approvals:
                    st.session_state.highlight_approvals[cid] = {}
                
                for filename, data in highlights.items():
                    if filename not in st.session_state.highlight_approvals[cid]:
                        st.session_state.highlight_approvals[cid][filename] = {}
                    for criterion_id, quotes in data.get('quotes', {}).items():
                        for i, quote in enumerate(quotes):
                            quote_key = quote['quote'][:100]
                            st.session_state.highlight_approvals[cid][filename][quote_key] = False
                            st.session_state[f"quote_{cid}_{filename}_{i}"] = False
                st.rerun()
        
        st.markdown("---")
        
        # Show highlights for each PDF
        for filename, data in highlights.items():
            render_pdf_highlights(cid, filename, data)
        
        # Regenerate option
        st.divider()
        st.markdown("### 🔄 Not satisfied with quotes?")
        
        feedback_text = st.text_area(
            "Tell AI what quotes to find",
            placeholder="e.g., 'Focus on award names and dates' or 'Need quotes about critical acclaim'",
            key=f"highlight_feedback_{cid}",
            height=60
        )
        
        if st.button("🔄 Regenerate Highlights", key=f"regen_highlights_{cid}"):
            regenerate_highlights(cid, feedback_text)
            st.rerun()


def render_pdf_highlights(cid: str, filename: str, data: dict):
    """Show highlights for a single PDF"""
    
    quotes_by_criterion = data.get('quotes', {})
    notes = data.get('notes', '')
    skip_highlighting = data.get('skip_highlighting', False)
    
    # Initialize approvals in session state if needed
    if cid not in st.session_state.highlight_approvals:
        st.session_state.highlight_approvals[cid] = {}
    if filename not in st.session_state.highlight_approvals[cid]:
        st.session_state.highlight_approvals[cid][filename] = {}
    
    # Read directly from session state
    file_approvals = st.session_state.highlight_approvals[cid][filename]
    
    # Count quotes
    total_quotes = sum(len(quotes) for quotes in quotes_by_criterion.values())
    approved_quotes = sum(1 for ok in file_approvals.values() if ok)
    
    # Show different header if skip_highlighting
    if skip_highlighting:
        st.markdown(f"**📄 {filename}** 🔒 _Included as-is (no highlighting)_")
    else:
        st.markdown(f"**📄 {filename}** ({total_quotes} quotes, {approved_quotes} approved)")
    
    if skip_highlighting:
        st.caption("✓ This document was marked to skip highlighting and will be included in the export without annotations.")
        st.markdown("---")
        return
    
    if total_quotes == 0:
        st.caption("No quotes found in this document")
        st.markdown("---")
        return
    
    # Show quotes by criterion
    for criterion_id, quotes in quotes_by_criterion.items():
        if not quotes:
            continue
        
        st.caption(f"**Criterion ({criterion_id}):** {len(quotes)} quote(s)")
        
        for i, quote_data in enumerate(quotes):
            quote_text = quote_data.get('quote', '')
            strength = quote_data.get('strength', 'medium')
            
            quote_key = quote_text[:100]  # Use first 100 chars as key
            # Read directly from session state
            is_approved = st.session_state.highlight_approvals[cid][filename].get(quote_key, True)
            
            # Strength indicator
            strength_emoji = {
                'high': '🟢',
                'medium': '🟡',
                'low': '🔴'
            }.get(strength, '⚪')
            
            # Checkbox for quote
            new_approval = st.checkbox(
                f"{strength_emoji} [{strength}] \"{quote_text[:150]}{'...' if len(quote_text) > 150 else ''}\"",
                value=is_approved,
                key=f"quote_{cid}_{filename}_{i}"
            )
            # Write directly to session state
            st.session_state.highlight_approvals[cid][filename][quote_key] = new_approval
    
    if notes:
        with st.expander("📝 AI Notes"):
            st.caption(notes)
    
    st.markdown("---")


def highlight_all_criteria():
    """Highlight all PDFs across all criteria"""
    
    total_pdfs = sum(len(pdfs) for pdfs in st.session_state.criterion_pdfs.items())
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    processed = 0
    
    for cid in CRITERIA.keys():
        if cid in st.session_state.criterion_pdfs:
            status.text(f"Highlighting Criterion ({cid})...")
            highlight_criterion(cid, show_progress=False)
            processed += 1
            progress_bar.progress(processed / len(st.session_state.criterion_pdfs))
    
    progress_bar.progress(1.0)
    status.empty()
    st.success("✅ All criteria highlighted!")
    st.rerun()


def highlight_criterion(cid: str, show_progress: bool = True):
    """Highlight all PDFs in a criterion"""
    
    pdfs = st.session_state.criterion_pdfs.get(cid, {})
    
    if not pdfs:
        return
    
    from src.pdf_text import extract_text_from_pdf_bytes
    from src.openai_terms import suggest_ovisa_quotes
    
    if cid not in st.session_state.highlight_results:
        st.session_state.highlight_results[cid] = {}
    
    # Get skip_highlighting flags
    skip_flags = st.session_state.skip_highlighting.get(cid, {})
    
    if show_progress:
        progress_bar = st.progress(0)
        status = st.empty()
    
    processed = 0
    
    for filename, pdf_bytes in pdfs.items():
        # Check if this document should skip highlighting
        if skip_flags.get(filename, False):
            if show_progress:
                status.text(f"Skipping {filename} (marked as no highlighting)...")
            
            # Store as-is without highlighting
            st.session_state.highlight_results[cid][filename] = {
                'quotes': {},  # Empty quotes
                'notes': 'Document marked to skip highlighting - included as-is',
                'pdf_bytes': pdf_bytes,
                'skip_highlighting': True  # Flag for export
            }
            
            processed += 1
            if show_progress:
                progress_bar.progress(processed / len(pdfs))
            continue
        
        if show_progress:
            status.text(f"Analyzing {filename}...")
        
        try:
            # Extract text
            text = extract_text_from_pdf_bytes(pdf_bytes)
            
            # Get quotes
            result = suggest_ovisa_quotes(
                document_text=text,
                beneficiary_name=st.session_state.beneficiary_name,
                beneficiary_variants=st.session_state.beneficiary_variants,
                selected_criteria_ids=[cid],
                feedback=None,
                user_feedback_text=None
            )
            
            # Store results
            st.session_state.highlight_results[cid][filename] = {
                'quotes': result.get('by_criterion', {}),
                'notes': result.get('notes', ''),
                'pdf_bytes': pdf_bytes,
                'skip_highlighting': False
            }
        
        except Exception as e:
            st.error(f"Error highlighting {filename}: {str(e)}")
        
        processed += 1
        if show_progress:
            progress_bar.progress(processed / len(pdfs))
    
    if show_progress:
        progress_bar.progress(1.0)
        status.empty()
        st.success(f"✅ Highlighted {processed} PDF(s) in criterion ({cid})")


def regenerate_highlights(cid: str, feedback_text: str):
    """Regenerate highlights with user feedback"""
    
    pdfs = st.session_state.criterion_pdfs.get(cid, {})
    
    if not pdfs:
        return
    
    from src.pdf_text import extract_text_from_pdf_bytes
    from src.openai_terms import suggest_ovisa_quotes
    
    with st.spinner("Regenerating highlights..."):
        for filename, pdf_bytes in pdfs.items():
            try:
                text = extract_text_from_pdf_bytes(pdf_bytes)
                
                result = suggest_ovisa_quotes(
                    document_text=text,
                    beneficiary_name=st.session_state.beneficiary_name,
                    beneficiary_variants=st.session_state.beneficiary_variants,
                    selected_criteria_ids=[cid],
                    feedback=None,
                    user_feedback_text=feedback_text
                )
                
                st.session_state.highlight_results[cid][filename] = {
                    'quotes': result.get('by_criterion', {}),
                    'notes': result.get('notes', ''),
                    'pdf_bytes': pdf_bytes
                }
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.success("✅ Regenerated highlights!")


def render_export_section():
    """Export section at bottom of page"""
    
    st.subheader("📦 Export Evidence Package")
    
    # Count what's available
    total_highlights = sum(
        len(highlights)
        for highlights in st.session_state.highlight_results.values()
    )
    
    total_approved = 0
    for cid, file_dict in st.session_state.highlight_approvals.items():
        for filename, quote_dict in file_dict.items():
            total_approved += sum(1 for approved in quote_dict.values() if approved)
    
    if total_highlights == 0:
        st.info("Highlight PDFs above first, then export")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Highlighted PDFs", total_highlights)
    with col2:
        st.metric("Approved Quotes", total_approved)
    
    st.divider()
    
    # Export options
    package_name = st.text_input(
        "Package name",
        value=f"evidence_package_{datetime.now().strftime('%Y-%m-%d')}",
        help="Name of the ZIP file"
    )
    
    if st.button("📦 Download ZIP Package", type="primary", use_container_width=True):
        try:
            zip_bytes = generate_export_zip(package_name)
            
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_bytes,
                file_name=f"{package_name}.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        except Exception as e:
            st.error(f"Export error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    st.divider()
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back", key="nav_back_highlight", use_container_width=True):
            st.session_state["goto_tab"] = "research"
            st.rerun()
    with nav_col2:
        if st.button("Next Page →", key="nav_next_highlight", use_container_width=True):
            st.session_state["goto_tab"] = "beneficiary_docs"
            st.rerun()


def generate_export_zip(package_name: str) -> bytes:
    """Generate ZIP with criterion subfolders and ANNOTATED PDFs"""
    
    from src.pdf_highlighter import annotate_pdf_bytes
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Add annotated PDFs organized by criterion
        for cid, highlights in st.session_state.highlight_results.items():
            
            # Create folder name
            desc = CRITERIA.get(cid, "Unknown")
            folder_name = f"Criterion_{cid}_{get_short_descriptor(cid)}"
            
            for filename, data in highlights.items():
                pdf_bytes = data['pdf_bytes']
                quotes_dict = data.get('quotes', {})
                
                # Get approved quotes only
                approved_quotes = []
                if cid in st.session_state.highlight_approvals:
                    if filename in st.session_state.highlight_approvals[cid]:
                        file_approvals = st.session_state.highlight_approvals[cid][filename]
                        
                        # Collect all approved quotes from all criteria
                        for criterion_id, quote_list in quotes_dict.items():
                            for quote_data in quote_list:
                                quote_text = quote_data.get('quote', '')
                                quote_key = quote_text[:100]
                                
                                if file_approvals.get(quote_key, True):  # Default approve
                                    approved_quotes.append(quote_text)
                # Check if this document should skip highlighting
                skip_highlight = data.get('skip_highlighting', False)
                
                # If no approvals tracked, use all quotes
                if not approved_quotes and not skip_highlight:
                    for criterion_id, quote_list in quotes_dict.items():
                        for quote_data in quote_list:
                            approved_quotes.append(quote_data.get('quote', ''))
                
                # Handle skip_highlighting flag
                if skip_highlight:
                    # Include original PDF without annotation
                    zip_path = f"{package_name}/{folder_name}/{filename}"
                    zip_file.writestr(zip_path, pdf_bytes)
                    continue
                
                # Annotate PDF with approved quotes
                try:
                    # Try to extract metadata from the PDF or use stored metadata
                    from src.metadata import autodetect_metadata
                    from src.pdf_text import extract_text_from_pdf_bytes
                    from datetime import datetime
                    
                    # Extract text for metadata detection
                    pdf_text = extract_text_from_pdf_bytes(pdf_bytes)
                    
                    # Auto-detect metadata
                    detected_meta = autodetect_metadata(pdf_text)
                    
                    # Build metadata for annotations
                    meta = {
                        "source_url": detected_meta.get("source_url", ""),
                        "venue_name": detected_meta.get("venue_name", ""),
                        "ensemble_name": detected_meta.get("ensemble_name", ""),
                        "performance_date": detected_meta.get("performance_date", ""),
                        "beneficiary_name": st.session_state.beneficiary_name,
                        "beneficiary_variants": st.session_state.beneficiary_variants,
                    }
                    
                    annotated_pdf, stats = annotate_pdf_bytes(
                        pdf_bytes=pdf_bytes,
                        quote_terms=approved_quotes,
                        criterion_id=cid,
                        meta=meta,
                        current_date=datetime.now()  # For past/future detection
                    )

                    if cid not in st.session_state.annotated_pdfs:
                        st.session_state.annotated_pdfs[cid] = {}
                    st.session_state.annotated_pdfs[cid][filename] = annotated_pdf
                    
                    # Add annotated PDF to ZIP
                    zip_path = f"{package_name}/{folder_name}/{filename}"
                    zip_file.writestr(zip_path, annotated_pdf)
                
                except Exception as e:
                    # If annotation fails, use original PDF
                    st.warning(f"Could not annotate {filename}: {str(e)}")
                    zip_path = f"{package_name}/{folder_name}/{filename}"
                    zip_file.writestr(zip_path, pdf_bytes)
        
        # Add README
        readme = generate_readme(package_name)
        zip_file.writestr(f"{package_name}/README.txt", readme)
    
    zip_buffer.seek(0)
    return zip_buffer.read()


def get_short_descriptor(cid: str) -> str:
    """Get short folder name for criterion"""
    descriptors = {
        "1": "Awards",
        "2_past": "Past_Roles",
        "2_future": "Future_Roles",
        "3": "Reviews",
        "4_past": "Past_Orgs",
        "4_future": "Future_Orgs",
        "5": "Success",
        "6": "Recognition",
        "7": "Salary"
    }
    return descriptors.get(cid, cid.replace("_", "-"))


def generate_readme(package_name: str) -> str:
    """Generate README for package"""
    
    beneficiary = st.session_state.beneficiary_name
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    total_pdfs = sum(len(h) for h in st.session_state.highlight_results.values())
    total_quotes = sum(
        sum(len(quotes) for quotes in h.get('quotes', {}).values())
        for highlights in st.session_state.highlight_results.values()
        for h in highlights.values()
    )
    
    readme = f"""O-1 VISA EVIDENCE PACKAGE
Generated: {date}
Beneficiary: {beneficiary}

PACKAGE CONTENTS:
================

Total PDFs: {total_pdfs}
Total Highlighted Quotes: {total_quotes}

FOLDER STRUCTURE:
"""
    
    for cid, highlights in st.session_state.highlight_results.items():
        desc = CRITERIA.get(cid, "")
        folder = f"Criterion_{cid}_{get_short_descriptor(cid)}"
        
        readme += f"\n{folder}/\n"
        readme += f"  Criterion ({cid}): {desc}\n"
        readme += f"  Files: {len(highlights)}\n"
        
        for filename in highlights.keys():
            readme += f"    - {filename}\n"
    
    readme += f"""

USAGE:
======
1. Review highlighted PDFs in each criterion folder
2. Verify quotes meet USCIS requirements
3. Add supporting documentation as needed
4. Submit with O-1 petition

Generated by O-1 Visa Evidence Assistant
"""
    
    return readme
