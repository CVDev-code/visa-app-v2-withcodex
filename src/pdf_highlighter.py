import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7          # thickness for red rectangles
LINE_WIDTH = 1.6         # thickness for connectors
FONTNAME = "Times-Bold"  # thicker than Times-Roman
FONT_SIZES = [12, 11, 10, 9, 8]

# ---- spacing knobs ----
EDGE_PAD = 18.0
GAP_FROM_TEXT_BLOCKS = 10.0
GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 1.5   # pull line end slightly away from box interior

# For quote search robustness
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18


# ============================================================
# NEW: Fixed text area detection for your PDF format
# ============================================================

def _get_fixed_text_area(page: fitz.Page) -> fitz.Rect:
    """
    Define fixed text area based on your standard PDF layout.
    This creates the "no-go zone" for annotations.
    """
    page_rect = page.rect
    
    # Based on your sample PDF layout
    # Header area: ~80-100px from top
    # Footer area: ~60px from bottom  
    # Text appears to be centered with margins on both sides
    
    TEXT_AREA_TOP = 100.0      # Below header
    TEXT_AREA_BOTTOM = page_rect.height - 60.0  # Above footer
    TEXT_AREA_LEFT = page_rect.width * 0.15     # 15% from left
    TEXT_AREA_RIGHT = page_rect.width * 0.85    # 15% from right (leaves 70% for text)
    
    return fitz.Rect(TEXT_AREA_LEFT, TEXT_AREA_TOP, TEXT_AREA_RIGHT, TEXT_AREA_BOTTOM)


# ============================================================
# Geometry helpers (unchanged)
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad
    rr.y0 -= pad
    rr.x1 += pad
    rr.y1 += pad
    return rr


def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects:
        return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    """Sample points along segment; good enough for routing decisions."""
    steps = 22
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)):
            return True
    return False


def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)


def _mid_edge_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    """Mid-height on left or right edge of callout box."""
    y = callout.y0 + (callout.height / 2.0)
    cx = callout.x0 + callout.width / 2.0
    if toward.x >= cx:
        return fitz.Point(callout.x1, y)
    return fitz.Point(callout.x0, y)


def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
    """Edge points for sampling endpoints (midpoints + slight corner bias)."""
    cx = rect.x0 + rect.width / 2.0
    cy = rect.y0 + rect.height / 2.0
    return [
        fitz.Point(rect.x0, cy),
        fitz.Point(rect.x1, cy),
        fitz.Point(cx, rect.y0),
        fitz.Point(cx, rect.y1),
        fitz.Point(rect.x0, rect.y0),
        fitz.Point(rect.x1, rect.y0),
        fitz.Point(rect.x0, rect.y1),
        fitz.Point(rect.x1, rect.y1),
    ]


def _straight_connector_best_pair(
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Point]:
    """
    ALWAYS returns a straight segment (start,end) that minimizes crossings.
    We sample multiple start points along the callout side and multiple end points on target edge.
    """
    target_center = _center(target_rect)

    # sample a few start points around the mid-height to avoid cutting through adjacent boxes
    base = _mid_edge_anchor(callout_rect, target_center)
    nudges = [-14.0, -7.0, 0.0, 7.0, 14.0]
    starts: List[fitz.Point] = []
    for dy in nudges:
        y = min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)
        starts.append(fitz.Point(base.x, y))

    ends = _edge_candidates(target_rect)

    best = (10**9, 10**9, starts[0], ends[0])  # (hits, length, start, end)
    for s in starts:
        for e in ends:
            hits = 0
            for ob in obstacles:
                if _segment_hits_rect(s, e, ob):
                    hits += 1

            length = math.hypot(e.x - s.x, e.y - s.y)

            if hits < best[0] or (hits == best[0] and length < best[1]):
                best = (hits, length, s, e)

    s = best[2]
    e = best[3]
    e = _pull_back_point(s, e, ENDPOINT_PULLBACK)
    return s, e


def _draw_straight_connector(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    s, e = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)


# ============================================================
# Callout text wrapping (unchanged)
# ============================================================

def _wrap_words_to_width(
    words: List[str],
    box_width: float,
    fs: int,
    *,
    max_words_per_line: Optional[int] = None,
    hard_break_long_words: bool = True,
) -> List[str]:
    """Wrap words to width. Optionally cap words per line; optionally hard-break long words."""
    usable_w = max(10.0, box_width - 10.0)

    def fits(s: str) -> bool:
        return fitz.get_text_length(s, fontname=FONTNAME, fontsize=fs) <= usable_w

    lines: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal cur
        if cur:
            lines.append(" ".join(cur))
            cur = []

    for w in words:
        # If the word alone doesn't fit, optionally hard-break it
        if not fits(w) and hard_break_long_words:
            flush()
            chunk = ""
            for ch in w:
                t = chunk + ch
                if fits(t):
                    chunk = t
                else:
                    if chunk:
                        lines.append(chunk)
                        chunk = ch
                    else:
                        lines.append(ch)
                        chunk = ""
            if chunk:
                cur = [chunk]
            continue

        # Try to add to current line
        trial = cur + [w]
        if (max_words_per_line is not None) and (len(trial) > max_words_per_line):
            flush()
            cur = [w]
            continue

        if fits(" ".join(trial)):
            cur = trial
        else:
            flush()
            cur = [w]

    flush()
    return lines


def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    """
    Prefer wrapping over shrinking.
    Tries (for each font size): normal wrap -> max 2 words/line -> 1 word/line.
    Only if those fail does it move to smaller fonts (9/8).
    """
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    words = text.split()

    wrap_modes = [
        (None),  # normal wrap (no cap)
        (2),     # max 2 words per line
        (1),     # 1 word per line
    ]

    MAX_H = 140.0

    for fs in FONT_SIZES:
        for cap in wrap_modes:
            lines = _wrap_words_to_width(
                words,
                box_width,
                fs,
                max_words_per_line=cap,
                hard_break_long_words=True,
            )
            h = (len(lines) * fs * 1.22) + 10.0
            wrapped = "\n".join(lines)

            if h <= MAX_H or fs == FONT_SIZES[-1]:
                return fs, wrapped, box_width, h

    return FONT_SIZES[-1], text, box_width, 44.0


# ============================================================
# NEW: Fixed margin placement system
# ============================================================

def _choose_best_margin_spot_fixed(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied_callouts: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    """
    Place callouts ONLY in margins outside the fixed text area.
    """
    page_rect = page.rect
    text_area = _get_fixed_text_area(page)
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2
    
    # Margin dimensions
    MARGIN_WIDTH = 140.0  # Max width for callouts
    MIN_MARGIN_WIDTH = 60.0  # Minimum usable width
    EDGE_BUFFER = 8.0
    VERTICAL_GAP = 12.0
    
    # Get available margin space
    left_margin_width = text_area.x0 - EDGE_BUFFER
    right_margin_width = page_rect.width - text_area.x1 - EDGE_BUFFER
    
    # Determine which margins are usable
    usable_margins = []
    if left_margin_width >= MIN_MARGIN_WIDTH:
        usable_margins.append(("left", EDGE_BUFFER, min(EDGE_BUFFER + MARGIN_WIDTH, text_area.x0 - EDGE_BUFFER)))
    if right_margin_width >= MIN_MARGIN_WIDTH:
        usable_margins.append(("right", max(text_area.x1 + EDGE_BUFFER, page_rect.width - EDGE_BUFFER - MARGIN_WIDTH), page_rect.width - EDGE_BUFFER))
    
    # Prefer left margin for consistency
    usable_margins.sort(key=lambda x: 0 if x[0] == "left" else 1)
    
    if not usable_margins:
        # Fallback: tiny callout at page edge
        fallback_rect = fitz.Rect(10, target_y - 20, 50, target_y + 20)
        return fallback_rect, label[:20], 8, False
    
    # Try each usable margin
    best = None
    occupied_buf = [inflate_rect(o, VERTICAL_GAP) for o in occupied_callouts]
    
    for side, margin_x0, margin_x1 in usable_margins:
        margin_width = margin_x1 - margin_x0
        
        # Get optimal text layout for this width
        fs, wrapped_text, w_used, h_needed = _optimize_layout_for_margin(label, margin_width)
        
        # Position vertically centered on target
        y0 = target_y - h_needed / 2
        y1 = target_y + h_needed / 2
        
        # Keep within page bounds
        y0 = max(EDGE_BUFFER, y0)
        y1 = min(page_rect.height - EDGE_BUFFER, y1)
        
        # Create callout rectangle
        if side == "left":
            callout_rect = fitz.Rect(margin_x1 - w_used, y0, margin_x1, y1)
        else:  # right
            callout_rect = fitz.Rect(margin_x0, y0, margin_x0 + w_used, y1)
        
        # Check for conflicts
        conflicts = any(r.intersects(callout_rect) for r in occupied_buf)
        
        # Score this position (prefer non-conflicting, left side)
        score = (1000 if conflicts else 0) + (10 if side == "right" else 0)
        
        if best is None or score < best[0]:
            best = (score, callout_rect, wrapped_text, fs, not conflicts)
        
        # If we found a non-conflicting spot, use it
        if not conflicts:
            break
    
    if best:
        return best[1], best[2], best[3], best[4]
    
    # Last resort: force placement even with conflicts
    side, margin_x0, margin_x1 = usable_margins[0]
    margin_width = margin_x1 - margin_x0
    fs, wrapped_text, w_used, h_needed = _optimize_layout_for_margin(label, margin_width)
    
    y0 = target_y - h_needed / 2
    y1 = target_y + h_needed / 2
    y0 = max(EDGE_BUFFER, y0)
    y1 = min(page_rect.height - EDGE_BUFFER, y1)
    
    if side == "left":
        callout_rect = fitz.Rect(margin_x1 - w_used, y0, margin_x1, y1)
    else:
        callout_rect = fitz.Rect(margin_x0, y0, margin_x1, y1)
    
    return callout_rect, wrapped_text, fs, False


# ============================================================
# Search helpers (unchanged)
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    """
    Robust search:
    1) exact search (with dehyphenate)
    2) whitespace-normalized search
    3) chunk fallback for long phrases (line breaks/hyphenation)
    """
    t = (term or "").strip()
    if not t:
        return []

    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    flags = 0
    try:
        flags |= fitz.TEXT_DEHYPHENATE
    except Exception:
        pass
    try:
        flags |= fitz.TEXT_PRESERVE_WHITESPACE
    except Exception:
        pass

    try:
        rects = page.search_for(t, flags=flags)
        if rects:
            return rects
    except Exception:
        pass

    t2 = _normalize_spaces(t)
    if t2 and t2 != t:
        try:
            rects = page.search_for(t2, flags=flags)
            if rects:
                return rects
        except Exception:
            pass

    if len(t2) >= _CHUNK:
        hits: List[fitz.Rect] = []
        step = max(10, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t2), step):
            chunk = t2[i:i + _CHUNK].strip()
            if len(chunk) < 18:
                continue
            try:
                rs = page.search_for(chunk, flags=flags)
                for r in rs:
                    hits.append(r)
            except Exception:
                continue

        if hits:
            hits_sorted = sorted(hits, key=lambda r: (r.y0, r.x0))
            merged: List[fitz.Rect] = []
            for r in hits_sorted:
                if not merged:
                    merged.append(fitz.Rect(r))
                else:
                    last = merged[-1]
                    if last.intersects(r) or abs(last.y0 - r.y0) < 3.0:
                        merged[-1] = last | r
                    else:
                        merged.append(fitz.Rect(r))
            return merged

    return []


# ============================================================
# Main annotation entrypoint - UPDATED
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)

    total_quote_hits = 0
    total_meta_hits = 0
    occupied_callouts: List[fitz.Rect] = []

    # A) Quote highlights (all pages)
    for page in doc:
        for term in (quote_terms or []):
            rects = _search_term(page, term)
            for r in rects:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1

    # B) Metadata callouts (page 1) - UPDATED to use fixed placement
    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connect_policy: str = "union",  # "single" | "union" | "all"
        also_try_variants: Optional[List[str]] = None,
    ):
        nonlocal total_meta_hits

        needles: List[str] = []
        if value and str(value).strip():
            needles.append(str(value).strip())
        if also_try_variants:
            for v in also_try_variants:
                vv = (v or "").strip()
                if vv:
                    needles.append(vv)

        needles = list(dict.fromkeys(needles))
        if not needles:
            return

        targets: List[fitz.Rect] = []
        for needle in needles:
            try:
                ts = page1.search_for(needle)
            except Exception:
                ts = []
            if ts:
                targets.extend(ts)

        if not targets:
            return

        for t in targets:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_meta_hits += len(targets)

        # USE THE NEW FIXED PLACEMENT FUNCTION
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot_fixed(page1, targets, occupied_callouts, label)

        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,   # <-- critical
        )

        obstacles: List[fitz.Rect] = []
        try:
            for b in page1.get_text("blocks"):
                obstacles.append(inflate_rect(fitz.Rect(b[:4]), 1.5))
        except Exception:
            pass

        expanded_all = [inflate_rect(t, 2.5) for t in targets]

        def connect_to(rect: fitz.Rect):
            obs = obstacles[:]
            for ot in expanded_all:
                if not (ot.intersects(rect) and (ot | rect).get_area() < (ot.get_area() + rect.get_area() + 3.0)):
                    obs.append(ot)
            for oc in occupied_callouts:
                obs.append(inflate_rect(oc, 2.0))

            _draw_straight_connector(page1, callout_rect, rect, obs)

        if connect_policy == "all":
            for t in targets:
                connect_to(t)
        elif connect_policy == "single":
            connect_to(targets[0])
        else:
            connect_to(_union_rect(targets))

        occupied_callouts.append(callout_rect)

    _do_job("Original source of publication.", meta.get("source_url"), connect_policy="union")
    _do_job("The distinguished organization.", meta.get("venue_name") or meta.get("org_name"), connect_policy="union")
    _do_job("Performance date.", meta.get("performance_date"), connect_policy="union")
    _do_job("Beneficiary salary evidence.", meta.get("salary_amount"), connect_policy="union")
    _do_job(
        "Beneficiary lead role evidence.",
        meta.get("beneficiary_name"),
        connect_policy="all",
        also_try_variants=meta.get("beneficiary_variants") or [],
    )

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
