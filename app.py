"""
O-1 Visa Evidence Assistant
2-Tab Workflow: Research → Highlight & Export
"""

import streamlit as st
import streamlit.components.v1 as components
from src.prompts import CRITERIA

# Page config
st.set_page_config(
    page_title="O-1 Visa Evidence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "visa_type": "O",  # O or P
        "beneficiary_name": "",
        "beneficiary_variants": [],
        "artist_field": "",
        "beneficiaries": [],          # list of dicts (name + details)
        "beneficiary_groups": [],     # list of beneficiary chunks (max 25 for P)
        "beneficiary_docs": {},       # {beneficiary_name: {passports:[], other_docs:[]}}
        "beneficiary_details": {},    # {beneficiary_name: {...}}
        "forms_state": {              # standard + per-beneficiary overrides
            "standard": {},
            "per_beneficiary": {}
        },
        "cover_letter_template": "",
        "cover_letter_text": "",
        "credentials_results": {},   # mirror of research_results for credentials
        "credentials_approvals": {}, # {source_id: True/False}
        "final_packets": {},         # cached packet outputs
        
        # Tab 1: Research results by criterion
        "research_results": {},      # {cid: [{url, title, excerpt, source}, ...]}
        "research_approvals": {},    # {cid: {url: True/False, ...}}
        "skip_highlighting": {},     # {cid: {filename: True/False, ...}} - True = skip highlighting
        
        # Tab 2: PDFs and highlights by criterion  
        "criterion_pdfs": {},        # {cid: {filename: bytes, ...}}
        "highlight_results": {},     # {cid: {filename: {quotes: {...}, notes: ""}, ...}}
        "highlight_approvals": {},   # {cid: {filename: {quote_text: True/False, ...}}}
        "annotated_pdfs": {},        # {cid: {filename: bytes}}
        "goto_tab": None,            # "research" or "highlight" to trigger tab switch
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()

# App header
st.title("📄 O-1 Visa Evidence Assistant")

# Visa type + beneficiary inputs (always visible at top)
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.session_state.beneficiary_name = st.text_input(
        "Beneficiary Name",
        value=st.session_state.beneficiary_name,
        placeholder="e.g., Yo-Yo Ma",
        help="Enter the artist's full name"
    )

with col2:
    st.session_state.artist_field = st.text_input(
        "Field (optional)",
        value=st.session_state.artist_field,
        placeholder="e.g., Classical Music"
    )

with col3:
    st.session_state.visa_type = st.radio(
        "Visa Type",
        options=["O", "P"],
        horizontal=True,
        index=0 if st.session_state.visa_type == "O" else 1
    )

# Name variants (collapsible)
with st.expander("📝 Name Variants (optional)"):
    variants_text = st.text_area(
        "Enter name variants (one per line)",
        value="\n".join(st.session_state.beneficiary_variants),
        placeholder="e.g.:\nYo Yo Ma\nYoYo Ma",
        height=80
    )
    st.session_state.beneficiary_variants = [
        v.strip() for v in variants_text.split("\n") if v.strip()
    ]

if not st.session_state.beneficiary_name:
    st.info("👆 Please enter beneficiary name to begin")
    st.stop()

st.divider()

tab_definitions = [
    ("research", "🔍 Research & Gather Evidence"),
    ("highlight", "✨ Highlight & Export"),
    ("beneficiary_docs", "📁 Beneficiary Documents"),
    ("beneficiary_details", "🧾 Beneficiary Details"),
    ("forms", "🧩 Forms"),
    ("cover_letter", "📝 Cover Letter"),
    ("credentials", "🏅 Credentials"),
    ("final_package", "📦 Final Package")
]

tab_labels = [label for _, label in tab_definitions]
tabs = st.tabs(tab_labels)

# ============================================================
# TAB 1: RESEARCH & GATHER EVIDENCE
# ============================================================
with tabs[0]:
    from src.research_tab import render_research_tab
    render_research_tab()

# ============================================================
# TAB 2: HIGHLIGHT & EXPORT
# ============================================================
with tabs[1]:
    from src.highlight_tab import render_highlight_tab
    render_highlight_tab()

# ============================================================
# TAB 3: BENEFICIARY DOCUMENTS
# ============================================================
with tabs[2]:
    from src.beneficiary_docs_tab import render_beneficiary_docs_tab
    render_beneficiary_docs_tab()

# ============================================================
# TAB 4: BENEFICIARY DETAILS
# ============================================================
with tabs[3]:
    from src.beneficiary_details_tab import render_beneficiary_details_tab
    render_beneficiary_details_tab()

# ============================================================
# TAB 5: FORMS
# ============================================================
with tabs[4]:
    from src.forms_tab import render_forms_tab
    render_forms_tab()

# ============================================================
# TAB 6: COVER LETTER
# ============================================================
with tabs[5]:
    from src.cover_letter_tab import render_cover_letter_tab
    render_cover_letter_tab()

# ============================================================
# TAB 7: CREDENTIALS
# ============================================================
with tabs[6]:
    from src.credentials_tab import render_credentials_tab
    render_credentials_tab()

# ============================================================
# TAB 8: FINAL PACKAGE
# ============================================================
with tabs[7]:
    from src.final_package_tab import render_final_package_tab
    render_final_package_tab()


# Handle programmatic tab navigation (e.g., "Next Page" / "Back")
goto_tab = st.session_state.get("goto_tab")
if goto_tab:
    tab_key_to_index = {key: idx for idx, (key, _) in enumerate(tab_definitions)}
    tab_index = tab_key_to_index.get(goto_tab, 0)
    components.html(
        f"""
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        if (tabs.length > {tab_index}) {{
            tabs[{tab_index}].click();
        }}
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state["goto_tab"] = None
