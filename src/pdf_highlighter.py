import io
import math
import os
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import fitz  # PyMuPDF
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"
FONT_SIZES = [11, 10, 9, 8, 7]

# ---- footer no-go zone (page coordinates; PyMuPDF = top-left origin) ----
NO_GO_RECT = fitz.Rect(
    21.00,   # left
    816.00,  # top
    411.26,  # right
    830.00   # bottom
)

# ---- spacing knobs ----
EDGE_PAD = 18.0  # Distance from page edge for annotations
GAP_FROM_TEXT_BLOCKS = 24.0  # Gap between text block and annotations
GAP_FROM_HIGHLIGHTS = 10.0
GAP_BETWEEN_CALLOUTS = 8.0
ENDPOINT_PULLBACK = 1.5

# ---- NEW: Annotation improvement constants ----
MIN_ANNOTATION_SPACING = 25.0   # Minimum vertical gap between annotations
MAX_ANNOTATION_DRIFT = 50.0     # Max distance from ideal Y position
OVERLAP_TOLERANCE = 2.0         # Extra padding to detect overlaps

# Connector density knobs (avoid clutter)
MAX_CONNECTORS_PER_PAGE = 3
MAX_TOTAL_CONNECTORS = 12
MIN_CONNECTOR_VERTICAL_SPACING = 40.0
CONNECTOR_ANGLE_OFFSET = 6.0
MIN_ANGLE_LENGTH = 40.0

# Arrowhead (DISABLED by setting to 0)
ARROW_LEN = 0.0  # Changed from 9.0 to 0.0 to disable arrowheads
ARROW_HALF_WIDTH = 0.0  # Changed from 4.5 to 0.0

# For quote search robustness
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18


# ============================================================
# Date parsing and comparison utilities
# ============================================================

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string in various common formats.
    Returns a datetime object or None if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    # Common date formats to try
    formats = [
        "%B %d, %Y",      # January 25, 2026
        "%b %d, %Y",      # Jan 25, 2026
        "%Y-%m-%d",       # 2026-01-25
        "%d/%m/%Y",       # 25/01/2026
        "%m/%d/%Y",       # 01/25/2026
        "%d.%m.%Y",       # 25.01.2026
        "%d-%m-%Y",       # 25-01-2026
        "%Y/%m/%d",       # 2026/01/25
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def get_date_label(performance_date_str: str, current_date: Optional[datetime] = None) -> str:
    """
    Determine if a performance date is in the past or future.
    Returns appropriate label text.
    
    Args:
        performance_date_str: The performance date string from metadata
        current_date: The current date (defaults to today if None)
    
    Returns:
        Either "Past performance date." or "Future performance date."
        or "Performance date." if date cannot be parsed
    """
    if current_date is None:
        current_date = datetime.now()
    
    performance_date = parse_date(performance_date_str)
    
    if performance_date is None:
        # Cannot parse date, use generic label
        return "Performance date."
    
    # Compare dates (ignoring time component)
    perf_date_only = performance_date.date()
    current_date_only = current_date.date()
    
    if perf_date_only < current_date_only:
        return "Past performance date."
    elif perf_date_only > current_date_only:
        return "Future performance date."
    else:
        # Same day - treat as current
        return "Performance date."


def _get_secret(name: str):
    """
    Works on Streamlit Cloud (st.secrets) and locally (.env / env vars).
    """
    try:
        import streamlit as st  # noqa: F401
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def _summarize_positive_description(quote_term: str) -> Optional[str]:
    """
    Use the LLM to produce a 1-3 word positive description.
    Returns None if the model is unavailable or fails.
    """
    if not quote_term or not quote_term.strip():
        return None

    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        return None

    model = _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You summarize praise in arts reviews. Return ONLY 1 to 3 words, "
        "no punctuation, no quotes, no extra text."
    )
    user_prompt = f"Quote:\n{quote_term.strip()}\n\nReturn 1-3 words."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    cleaned = re.sub(r"[\"'`.,;:!?\(\)\[\]\{\}]+", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    words = cleaned.split(" ")
    cleaned = " ".join(words[:3])
    return cleaned if cleaned else None


# ============================================================
# Geometry helpers
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


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _edge_points_for_rect(rect: fitz.Rect, toward: fitz.Point) -> List[fitz.Point]:
    """
    Candidate points on rect edges, biased toward a point.
    """
    x = _clamp(toward.x, rect.x0, rect.x1)
    y = _clamp(toward.y, rect.y0, rect.y1)
    return [
        fitz.Point(rect.x0, y),  # left
        fitz.Point(rect.x1, y),  # right
        fitz.Point(x, rect.y0),  # top
        fitz.Point(x, rect.y1),  # bottom
    ]


def _offset_point_outside_rect(p: fitz.Point, rect: fitz.Rect, pad: float = 1.5) -> fitz.Point:
    """
    Nudge a point so it sits just outside the rect edge.
    """
    if abs(p.x - rect.x0) < 0.1:
        return fitz.Point(p.x - pad, p.y)
    if abs(p.x - rect.x1) < 0.1:
        return fitz.Point(p.x + pad, p.y)
    if abs(p.y - rect.y0) < 0.1:
        return fitz.Point(p.x, p.y - pad)
    if abs(p.y - rect.y1) < 0.1:
        return fitz.Point(p.x, p.y + pad)
    return p


def _angle_offset_for_length(length: float) -> float:
    return max(4.0, min(12.0, length / 20.0))


def _choose_callout_margin_side(callout_rect: fitz.Rect, page_rect: fitz.Rect) -> Tuple[bool, float, float]:
    left_margin_x = EDGE_PAD
    right_margin_x = page_rect.width - EDGE_PAD
    dist_left = max(0.0, callout_rect.x0 - left_margin_x)
    dist_right = max(0.0, right_margin_x - callout_rect.x1)
    callout_left = dist_left <= dist_right
    return callout_left, left_margin_x, right_margin_x


def _choose_best_target_on_page(
    *,
    page: fitz.Page,
    candidates: List[fitz.Rect],
    margin_x: float,
    callout_left: bool,
    avoid_rects: List[fitz.Rect],
) -> Optional[fitz.Rect]:
    if not candidates:
        return None

    best = None
    best_hits = None
    best_len = None

    ordered = sorted(candidates, key=lambda r: (r.y0, r.x0))
    for r in ordered:
        end_x = r.x0 if callout_left else r.x1
        end_y = (r.y0 + r.y1) / 2.0
        length = abs(end_x - margin_x)

        offset = _angle_offset_for_length(length) if length > MIN_ANGLE_LENGTH else 0.0
        approach_y = max(EDGE_PAD + 2.0, end_y - offset) if offset > 0 else end_y

        start = fitz.Point(margin_x, approach_y)
        end = fitz.Point(end_x, end_y)

        hits = 0
        for obs in avoid_rects:
            if obs is r:
                continue
            if _segment_hits_rect(start, end, inflate_rect(obs, 1.5)):
                hits += 1

        if best is None or hits < best_hits or (hits == best_hits and length < best_len):
            best = r
            best_hits = hits
            best_len = length

        if hits == 0:
            break

    return best if best is not None else ordered[0]


def _compute_trunk_start(callout_rect: fitz.Rect, page_rect: fitz.Rect) -> Tuple[fitz.Point, bool, float]:
    callout_left, left_margin_x, right_margin_x = _choose_callout_margin_side(
        callout_rect, page_rect
    )
    start_x = callout_rect.x0 - 1.5 if callout_left else callout_rect.x1 + 1.5
    start_y = min(callout_rect.y1 - 1.0, page_rect.height - EDGE_PAD - 2.0)
    start = fitz.Point(start_x, start_y)
    margin_x = left_margin_x if callout_left else right_margin_x
    return start, callout_left, margin_x


def _end_point_from_start(start: fitz.Point, target_rect: fitz.Rect) -> fitz.Point:
    candidates = _edge_points_for_rect(target_rect, start)
    end = min(candidates, key=lambda p: math.hypot(p.x - start.x, p.y - start.y))

    if abs(end.y - start.y) < 1.0 and abs(end.x - start.x) > MIN_ANGLE_LENGTH:
        offset = _angle_offset_for_length(abs(end.x - start.x))
        desired_end_y = min(target_rect.y1 - 1.0, end.y + offset)
        if desired_end_y > start.y + 0.5:
            end = fitz.Point(end.x, desired_end_y)

    return end


def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)


def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect, steps: int = 60) -> bool:
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)):
            return True
    return False


def _shift_rect_up(rect: fitz.Rect, shift: float, min_y: float = 2.0) -> fitz.Rect:
    if shift <= 0:
        return fitz.Rect(rect)
    h = rect.y1 - rect.y0
    new_y1 = max(min_y + h, rect.y1 - shift)
    return fitz.Rect(rect.x0, new_y1 - h, rect.x1, new_y1)


# ============================================================
# HARD SAFETY: never pass invalid rects into insert_textbox
# ============================================================

def _rect_is_valid(r: fitz.Rect) -> bool:
    vals = [r.x0, r.y0, r.x1, r.y1]
    return (
        all(math.isfinite(v) for v in vals)
        and (r.x1 > r.x0)
        and (r.y1 > r.y0)
    )


def _ensure_min_size(
    r: fitz.Rect,
    pr: fitz.Rect,
    min_w: float = 20.0,
    min_h: float = 12.0,
    pad: float = 2.0,
) -> fitz.Rect:
    rr = fitz.Rect(r)

    cx = (rr.x0 + rr.x1) / 2.0
    cy = (rr.y0 + rr.y1) / 2.0
    w = max(min_w, abs(rr.x1 - rr.x0))
    h = max(min_h, abs(rr.y1 - rr.y0))

    rr = fitz.Rect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    rr.x0 = max(pad, rr.x0)
    rr.y0 = max(pad, rr.y0)
    rr.x1 = min(pr.width - pad, rr.x1)
    rr.y1 = min(pr.height - pad, rr.y1)

    if rr.x1 <= rr.x0 or rr.y1 <= rr.y0:
        rr = fitz.Rect(pad, pad, pad + min_w, pad + min_h)

    return rr


# ============================================================
# Text area detection (dynamic margins)
# ============================================================

def _get_fallback_text_area(page: fitz.Page) -> fitz.Rect:
    pr = page.rect
    return fitz.Rect(
        pr.width * 0.12,
        pr.height * 0.12,
        pr.width * 0.88,
        pr.height * 0.88,
    )


def _detect_actual_text_area(page: fitz.Page) -> fitz.Rect:
    try:
        words = page.get_text("words") or []
        if not words:
            return _get_fallback_text_area(page)

        pr = page.rect
        header_limit = pr.height * 0.12
        footer_limit = pr.height * 0.88

        x0s, x1s = [], []
        for w in words:
            x0, y0, x1, y1, text = w[:5]
            if y0 > header_limit and y1 < footer_limit and len((text or "").strip()) > 1:
                x0s.append(float(x0))
                x1s.append(float(x1))

        if not x0s:
            return _get_fallback_text_area(page)

        x0s.sort()
        x1s.sort()

        li = int(len(x0s) * 0.05)
        ri = int(len(x1s) * 0.95)

        text_left = x0s[max(0, li)]
        text_right = x1s[min(len(x1s) - 1, ri)]

        text_left = max(pr.width * 0.08, text_left)
        text_right = min(pr.width * 0.92, text_right)

        if text_right <= text_left + 50:
            return _get_fallback_text_area(page)

        return fitz.Rect(text_left, header_limit, text_right, footer_limit)
    except Exception:
        return _get_fallback_text_area(page)


# ============================================================
# Text wrapping (simple + reliable)
# ============================================================

def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    words = text.split()
    max_h = 180.0

    for fs in FONT_SIZES:
        usable_w = max(20.0, box_width - 10.0)
        lines: List[str] = []
        cur = ""

        for w in words:
            trial = (cur + " " + w).strip() if cur else w
            if fitz.get_text_length(trial, fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w

        if cur:
            lines.append(cur)

        wrapped = "\n".join(lines)
        h = (len(lines) * fs * 1.25) + 10.0

        if h <= max_h or fs == FONT_SIZES[-1]:
            return fs, wrapped, box_width, h

    return FONT_SIZES[-1], text, box_width, 44.0


# ============================================================
# Fit-guaranteed textbox insertion
# ============================================================

def _insert_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontname: str,
    fontsize: int,
    color,
    align=fitz.TEXT_ALIGN_LEFT,
    overlay: bool = True,
    max_expand_iters: int = 8,
    extra_pad_each_iter: float = 6.0,
) -> Tuple[fitz.Rect, float, int]:
    pr = page.rect
    r = fitz.Rect(rect)
    fs = int(fontsize)

    r = _ensure_min_size(r, pr)
    if not _rect_is_valid(r):
        return r, 0.0, fs

    attempt = 0
    while attempt < max_expand_iters:
        ret_val = page.insert_textbox(
            r,
            text,
            fontname=fontname,
            fontsize=fs,
            color=color,
            align=align,
            overlay=overlay,
        )

        if ret_val >= 0:
            return r, ret_val, fs

        attempt += 1
        r.y1 = min(r.y1 + extra_pad_each_iter, pr.height - 2.0)
        r.x1 = min(r.x1 + extra_pad_each_iter, pr.width - 2.0)

        if not _rect_is_valid(r):
            break

    # fallback
    final_ret = page.insert_textbox(
        r, text, fontname=fontname, fontsize=fs, color=color, align=align, overlay=overlay
    )
    return r, final_ret, fs


# ============================================================
# Search helpers (chunked, robust)
# ============================================================

def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    term = (term or "").strip()
    if not term:
        return []

    if len(term) <= _MAX_TERM:
        try:
            return page.search_for(term)
        except Exception:
            return []

    found_rects: List[fitz.Rect] = []
    length = len(term)
    start = 0

    while start < length:
        end = min(start + _CHUNK, length)
        chunk = term[start:end]

        try:
            rects = page.search_for(chunk)
            found_rects.extend(rects)
        except Exception:
            pass

        start += (_CHUNK - _CHUNK_OVERLAP)

    return found_rects


def _dedupe_rects(rects: List[fitz.Rect], pad: float = 1.0) -> List[fitz.Rect]:
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    out: List[fitz.Rect] = [rects[0]]

    for r in rects[1:]:
        merged = False
        for i, existing in enumerate(out):
            if inflate_rect(existing, pad).intersects(r):
                out[i] = existing | r
                merged = True
                break
        if not merged:
            out.append(r)

    return out


def _merge_rects_per_line(
    rects: List[fitz.Rect],
    *,
    y_tol: float = 2.0,
    x_gap: float = 3.0,
) -> List[fitz.Rect]:
    """
    Merge adjacent rects on the same line into a single box.
    """
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    lines: List[List[fitz.Rect]] = []

    for r in rects:
        y_mid = (r.y0 + r.y1) / 2.0
        placed = False
        for group in lines:
            gy = (group[0].y0 + group[0].y1) / 2.0
            if abs(y_mid - gy) <= y_tol:
                group.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])

    merged: List[fitz.Rect] = []
    for group in lines:
        group = sorted(group, key=lambda r: r.x0)
        cur = fitz.Rect(group[0])
        for r in group[1:]:
            if r.x0 <= cur.x1 + x_gap:
                cur |= r
            else:
                merged.append(cur)
                cur = fitz.Rect(r)
        merged.append(cur)

    return merged


# ============================================================
# Annotation placement (improved spacing)
# ============================================================

def _place_annotation_in_margin(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label_text: str,
    left_count: int,
    right_count: int,
) -> Tuple[fitz.Rect, str, int, bool]:
    """
    Places annotation in margin with improved vertical spacing.
    Returns: (callout_rect, wrapped_text, fontsize, is_safe_placement)
    """
    if not targets:
        pr = page.rect
        return fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + 100, EDGE_PAD + 40), label_text, 10, False

    target_union = _union_rect(targets)
    text_area = _detect_actual_text_area(page)
    pr = page.rect

    # Determine side based on balance
    if left_count <= right_count:
        # Place on left
        side = "left"
        margin_x0 = EDGE_PAD
        margin_x1 = text_area.x0 - GAP_FROM_TEXT_BLOCKS
    else:
        # Place on right
        side = "right"
        margin_x0 = text_area.x1 + GAP_FROM_TEXT_BLOCKS
        margin_x1 = pr.width - EDGE_PAD

    box_width = max(20.0, margin_x1 - margin_x0)
    fs, wrapped, _w, box_h = _optimize_layout_for_margin(label_text, box_width)

    # Ideal Y position (aligned with target)
    ideal_y = target_union.y0

    # Find safe Y position avoiding overlaps
    test_rect = fitz.Rect(margin_x0, ideal_y, margin_x1, ideal_y + box_h)
    
    # Check for overlaps with existing annotations
    def has_overlap(rect: fitz.Rect) -> bool:
        for occ in occupied:
            if inflate_rect(occ, MIN_ANNOTATION_SPACING).intersects(rect):
                return True
        return False
    
    # Try ideal position first
    if not has_overlap(test_rect):
        return test_rect, wrapped, fs, True
    
    # Try moving down in small increments
    max_drift = MAX_ANNOTATION_DRIFT
    step = 5.0
    for offset in range(0, int(max_drift), int(step)):
        test_rect = fitz.Rect(margin_x0, ideal_y + offset, margin_x1, ideal_y + offset + box_h)
        if test_rect.y1 > pr.height - EDGE_PAD:
            break
        if not has_overlap(test_rect):
            return test_rect, wrapped, fs, True
    
    # Try moving up
    for offset in range(int(step), int(max_drift), int(step)):
        test_rect = fitz.Rect(margin_x0, ideal_y - offset, margin_x1, ideal_y - offset + box_h)
        if test_rect.y0 < EDGE_PAD:
            break
        if not has_overlap(test_rect):
            return test_rect, wrapped, fs, True
    
    # Fallback: place at bottom of occupied stack
    if occupied:
        last_occ = max(occupied, key=lambda r: r.y1)
        fallback_y = last_occ.y1 + MIN_ANNOTATION_SPACING
        if fallback_y + box_h < pr.height - EDGE_PAD:
            test_rect = fitz.Rect(margin_x0, fallback_y, margin_x1, fallback_y + box_h)
            return test_rect, wrapped, fs, False
    
    # Ultimate fallback
    return fitz.Rect(margin_x0, ideal_y, margin_x1, ideal_y + box_h), wrapped, fs, False


# ============================================================
# Connector line routing
# ============================================================

def _edge_to_edge_points(r1: fitz.Rect, r2: fitz.Rect) -> Tuple[fitz.Point, fitz.Point]:
    """
    Determine optimal connection points between two rectangles.
    Improved logic: prioritizes horizontal connections and cleaner angles.
    """
    c1 = _center(r1)
    c2 = _center(r2)

    # Choose start/end points that minimize line length
    best_p1 = None
    best_p2 = None
    best_d = None

    for p1 in _edge_points_for_rect(r1, c2):
        p1 = _offset_point_outside_rect(p1, r1)
        candidates = _edge_points_for_rect(r2, p1)
        # Pick the closest target-edge point to this start
        p2 = min(candidates, key=lambda p: math.hypot(p.x - p1.x, p.y - p1.y))
        d = math.hypot(p2.x - p1.x, p2.y - p1.y)
        if best_d is None or d < best_d:
            best_d = d
            best_p1 = p1
            best_p2 = p2

    p1, p2 = best_p1, best_p2

    # Only add angle for long horizontals; always slope downward
    if abs(p1.y - p2.y) < 1.0 and abs(p2.x - p1.x) > MIN_ANGLE_LENGTH:
        offset = _angle_offset_for_length(abs(p2.x - p1.x))
        desired_end_y = min(r2.y1 - 1.0, p2.y + offset)
        if desired_end_y > p1.y + 0.5:
            p2 = fitz.Point(p2.x, desired_end_y)
        else:
            new_start_y = max(r1.y0 + 1.0, p1.y - offset)
            if p2.y > new_start_y + 0.5:
                p1 = fitz.Point(p1.x, new_start_y)

    return p1, p2


def _draw_arrowhead(page: fitz.Page, from_pt: fitz.Point, to_pt: fitz.Point):
    """
    Draw an arrowhead at to_pt pointing from from_pt.
    DISABLED: Returns immediately if ARROW_LEN is 0.
    """
    if ARROW_LEN == 0:
        return  # Arrowheads disabled
    
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return

    ux, uy = vx / d, vy / d
    base = fitz.Point(to_pt.x + ux * ARROW_LEN, to_pt.y + uy * ARROW_LEN)

    perp_x, perp_y = -uy, ux
    left = fitz.Point(base.x + perp_x * ARROW_HALF_WIDTH, base.y + perp_y * ARROW_HALF_WIDTH)
    right = fitz.Point(base.x - perp_x * ARROW_HALF_WIDTH, base.y - perp_y * ARROW_HALF_WIDTH)

    page.draw_polyline([left, to_pt, right], color=RED, fill=RED, width=0.5, closePath=True)


def _draw_routed_line(
    page: fitz.Page,
    start: fitz.Point,
    end: fitz.Point,
    obstacles: List[fitz.Rect],
):
    """
    Draw a line from start to end, routing around obstacles with right angles.
    Uses improved routing logic.
    """
    s = _pull_back_point(end, start, ENDPOINT_PULLBACK)
    e = _pull_back_point(start, end, ENDPOINT_PULLBACK)

    obstacles = [obs for obs in obstacles if not obs.contains(start) and not obs.contains(end)]

    # Check if direct path is clear
    direct_blocked = any(_segment_hits_rect(s, e, inflate_rect(obs, 2.0)) for obs in obstacles)
    
    if not direct_blocked:
        # Direct path is clear
        page.draw_line(s, e, color=RED, width=LINE_WIDTH)
        if ARROW_LEN > 0:  # Only draw arrowhead if enabled
            _draw_arrowhead(page, s, e)
        return

    # Need to route around obstacles
    # Try two-segment route (horizontal then vertical, or vice versa)
    mid_h_first = fitz.Point(e.x, s.y)  # horizontal first
    mid_v_first = fitz.Point(s.x, e.y)  # vertical first
    
    # Check horizontal-first route
    h_first_blocked = (
        any(_segment_hits_rect(s, mid_h_first, inflate_rect(obs, 2.0)) for obs in obstacles) or
        any(_segment_hits_rect(mid_h_first, e, inflate_rect(obs, 2.0)) for obs in obstacles)
    )
    
    # Check vertical-first route
    v_first_blocked = (
        any(_segment_hits_rect(s, mid_v_first, inflate_rect(obs, 2.0)) for obs in obstacles) or
        any(_segment_hits_rect(mid_v_first, e, inflate_rect(obs, 2.0)) for obs in obstacles)
    )
    
    if not h_first_blocked:
        # Use horizontal-first route
        page.draw_line(s, mid_h_first, color=RED, width=LINE_WIDTH)
        page.draw_line(mid_h_first, e, color=RED, width=LINE_WIDTH)
        if ARROW_LEN > 0:
            _draw_arrowhead(page, mid_h_first, e)
    elif not v_first_blocked:
        # Use vertical-first route
        page.draw_line(s, mid_v_first, color=RED, width=LINE_WIDTH)
        page.draw_line(mid_v_first, e, color=RED, width=LINE_WIDTH)
        if ARROW_LEN > 0:
            _draw_arrowhead(page, mid_v_first, e)
    else:
        # Both routes blocked, use direct path anyway
        page.draw_line(s, e, color=RED, width=LINE_WIDTH)
        if ARROW_LEN > 0:
            _draw_arrowhead(page, s, e)


def _draw_multipage_connector(
    doc: fitz.Document,
    callout_page_idx: int,
    callout_rect: fitz.Rect,
    target_page_idx: int,
    target_rect: fitz.Rect,
    *,
    occupied_callouts: Optional[List[fitz.Rect]] = None,
    last_target_page_idx: Optional[int] = None,
):
    """
    Draw a connector from a callout on one page to a target on another page.
    Routes through page margins.
    """
    if callout_page_idx == target_page_idx:
        return

    callout_page = doc.load_page(callout_page_idx)
    target_page = doc.load_page(target_page_idx)

    # Start from bottom edge on nearest margin-facing side
    start, callout_left, margin_x = _compute_trunk_start(callout_rect, callout_page.rect)

    # Choose a consistent margin side based on callout position
    if callout_left:
        end_x = target_rect.x0
    else:
        end_x = target_rect.x1

    # End at target (offset to avoid perfectly horizontal lines)
    target_center = _center(target_rect)
    end = fitz.Point(end_x, target_center.y)
    if abs(end.y - start.y) < 1.0 and abs(end.x - start.x) > MIN_ANGLE_LENGTH:
        offset = _angle_offset_for_length(abs(end.x - start.x))
        desired_end_y = min(target_rect.y1 - 1.0, end.y + offset)
        if desired_end_y > start.y + 0.5:
            end = fitz.Point(end.x, desired_end_y)

    # Draw vertical line to bottom of callout page
    obstacles = occupied_callouts or []
    y = start.y
    if obstacles:
        for offset in (0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0, 24.0, -24.0, 30.0, -30.0):
            y_try = start.y + offset
            seg = fitz.Rect(min(start.x, margin_x), y_try - 1.0, max(start.x, margin_x), y_try + 1.0)
            if not any(seg.intersects(inflate_rect(o, 2.0)) for o in obstacles if o != callout_rect):
                y = y_try
                break

    bottom_point = fitz.Point(margin_x, callout_page.rect.height - EDGE_PAD)

    callout_page.draw_line(start, fitz.Point(margin_x, y), color=RED, width=LINE_WIDTH)
    callout_page.draw_line(fitz.Point(margin_x, y), bottom_point, color=RED, width=LINE_WIDTH)

    # Draw on intermediate pages if any
    for pi in range(callout_page_idx + 1, target_page_idx):
        p = doc.load_page(pi)
        top_point = fitz.Point(margin_x, EDGE_PAD)
        bottom_point = fitz.Point(margin_x, p.rect.height - EDGE_PAD)
        p.draw_line(top_point, bottom_point, color=RED, width=LINE_WIDTH)

    # Draw on target page
    top_point = fitz.Point(margin_x, EDGE_PAD)
    bottom_point = fitz.Point(margin_x, target_page.rect.height - EDGE_PAD)
    if last_target_page_idx is None:
        last_target_page_idx = target_page_idx

    if target_page_idx < last_target_page_idx:
        # Continue trunk only if there are more pages
        target_page.draw_line(top_point, bottom_point, color=RED, width=LINE_WIDTH)
        branch_start = top_point
    else:
        # Last page: branch starts at trunk entry point only
        branch_start = top_point

    target_page.draw_line(branch_start, end, color=RED, width=LINE_WIDTH)
    
    # Draw arrowhead only if enabled
    if ARROW_LEN > 0:
        _draw_arrowhead(target_page, fitz.Point(margin_x, end.y), end)


def _select_targets_for_connectors(
    targets_by_page: Dict[int, List[fitz.Rect]],
    *,
    policy: str,
) -> Dict[int, List[fitz.Rect]]:
    """
    Select connector targets to reduce clutter while keeping coverage.
    """
    selected: Dict[int, List[fitz.Rect]] = {}
    total = 0

    if policy == "page_first":
        for pi in sorted(targets_by_page.keys()):
            rects = sorted(targets_by_page[pi], key=lambda r: (r.y0, r.x0))
            if rects:
                selected[pi] = [rects[0]]
                total += 1
            if total >= MAX_TOTAL_CONNECTORS:
                break
        return selected

    # policy == "all"
    for pi in sorted(targets_by_page.keys()):
        rects = sorted(targets_by_page[pi], key=lambda r: (r.y0, r.x0))
        page_selected: List[fitz.Rect] = []
        last_y: Optional[float] = None

        for r in rects:
            if last_y is not None and abs(r.y0 - last_y) < MIN_CONNECTOR_VERTICAL_SPACING:
                continue

            page_selected.append(r)
            last_y = r.y0

            if len(page_selected) >= MAX_CONNECTORS_PER_PAGE:
                break

        if page_selected:
            selected[pi] = page_selected
            total += len(page_selected)

        if total >= MAX_TOTAL_CONNECTORS:
            break

    return selected


# ============================================================
# Main annotation function
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
    current_date: Optional[datetime] = None,
) -> Tuple[bytes, Dict]:
    """
    Annotate a PDF with highlights and metadata callouts.
    
    Args:
        pdf_bytes: Input PDF as bytes
        quote_terms: List of text snippets to highlight (these get RED BOXES AND criterion-specific annotation)
        criterion_id: Identifier for the criterion (e.g., "criterion-1", "criterion-2", etc.)
        meta: Metadata dictionary containing:
            - source_url: URL of the publication
            - venue_name: Name of the venue/organization
            - ensemble_name: Name of the performing ensemble
            - performance_date: Date of the performance
            - beneficiary_name: Name of the beneficiary
            - beneficiary_variants: Alternative names for beneficiary
        current_date: Current date for date comparison (defaults to today)
    
    Returns:
        Tuple of (annotated_pdf_bytes, statistics_dict)
    
    ANNOTATION LABEL MAPPING:
    ========================
    Criterion 1:
      - "Original source of publication." -> meta["source_url"]
      - "Award issuer." -> meta["venue_name"]
      - Quote annotation: "Beneficiary receives award."

    Criterion 2:
      - "Original source of publication." -> meta["source_url"]
      - "past performance" or "future performance" -> meta["performance_date"]

    Criterion 3:
      - "Original source of publication." -> meta["source_url"]
      - Quote annotation: "Beneficiary's performance described as {1-3 words}"

    Criterion 4:
      - "Original source of publication." -> meta["source_url"]
      - "Distinguished organization." -> meta["venue_name"]
      - "Distinguished organization." -> meta["ensemble_name"]
      - "past performance" or "future performance" -> meta["performance_date"]

    Criterion 5:
      - "Original source of publication." -> meta["source_url"]
      - Quote annotation: "Beneficiary's successes are critically acclaimed"

    Criterion 6:
      - "Original source of publication." -> meta["source_url"]
      - Quote annotation: "Beneficiary's achievements received recognition from industry experts"

    Criterion 7:
      - "Original source of publication." -> meta["source_url"]

    NOTE: quote_terms get red boxes; the first quote_term gets the quote annotation.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)

    total_quote_hits = 0
    total_meta_hits = 0
    occupied_callouts: List[fitz.Rect] = []
    left_annotation_count = 0
    right_annotation_count = 0

    # Track quote hits with page index for multi-page connectors
    quote_hits_by_page: Dict[int, List[fitz.Rect]] = {}

    # A) Quote highlights (all pages) + dedupe per page
    # Track quote term occurrences for criterion-specific annotation connectors
    first_quote_term = quote_terms[0] if quote_terms else None
    quote_targets_by_term: Dict[str, Dict[int, List[fitz.Rect]]] = {}
    
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_hits: List[fitz.Rect] = []

        for term in (quote_terms or []):
            rects = _search_term(page, term)
            page_hits.extend(rects)
            if rects:
                quote_targets_by_term.setdefault(term, {}).setdefault(page_index, []).extend(rects)

        page_hits = _merge_rects_per_line(page_hits, y_tol=2.0, x_gap=3.0)
        page_hits = _dedupe_rects(page_hits, pad=1.0)
        if page_hits:
            quote_hits_by_page[page_index] = page_hits

        for r in page_hits:
            page.draw_rect(r, color=RED, width=BOX_WIDTH)
            total_quote_hits += 1
    
    # Deduplicate quote term targets per page per term
    for term, targets_by_page in list(quote_targets_by_term.items()):
        for pi in list(targets_by_page.keys()):
            targets_by_page[pi] = _merge_rects_per_line(targets_by_page[pi], y_tol=2.0, x_gap=3.0)
            targets_by_page[pi] = _dedupe_rects(targets_by_page[pi], pad=1.0)

    # B) Metadata callouts (page 1) — targets can exist on any page now
    connectors_to_draw = []  # list of dicts

    def _find_targets_across_doc(needle: str) -> List[Tuple[int, fitz.Rect]]:
        out: List[Tuple[int, fitz.Rect]] = []
        if not needle.strip():
            return out

        for pi in range(doc.page_count):
            p = doc.load_page(pi)
            try:
                rects = p.search_for(needle)
            except Exception:
                rects = []
            for r in rects:
                out.append((pi, r))
        return out

    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connect_policy: str = "union",  # "single" | "union" | "all"
        also_try_variants: Optional[List[str]] = None,
    ):
        nonlocal total_meta_hits
        nonlocal left_annotation_count
        nonlocal right_annotation_count

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

        # Find targets across ALL pages (then dedupe per page)
        targets_by_page: Dict[int, List[fitz.Rect]] = {}
        for needle in needles:
            hits = _find_targets_across_doc(needle)
            for pi, r in hits:
                targets_by_page.setdefault(pi, []).append(r)

        # Deduplicate per page
        cleaned_targets_by_page: Dict[int, List[fitz.Rect]] = {}
        for pi, rects in targets_by_page.items():
            deduped = _dedupe_rects(rects, pad=1.0)
            if deduped:
                cleaned_targets_by_page[pi] = deduped

        if not cleaned_targets_by_page:
            return

        # Box only FIRST occurrence, avoid double-boxing with quotes
        boxed_any = False
        
        # Try all occurrences in order until we box one
        for pi in sorted(cleaned_targets_by_page.keys()):
            if boxed_any:
                break
            
            p = doc.load_page(pi)
            page_quote_boxes = quote_hits_by_page.get(pi, [])
            
            for r in cleaned_targets_by_page[pi]:
                # Check if overlaps any quote box (quotes take priority)
                overlaps_quote = any(
                    r.intersects(inflate_rect(qr, OVERLAP_TOLERANCE)) 
                    for qr in page_quote_boxes
                )
                
                if not overlaps_quote:
                    # This is the first non-overlapping occurrence - box it
                    p.draw_rect(r, color=RED, width=BOX_WIDTH)
                    total_meta_hits += 1
                    
                    # Keep ONLY this occurrence for connector (don't overwrite dict structure!)
                    # Clear all other pages/rects, keep only this one
                    cleaned_targets_by_page.clear()
                    cleaned_targets_by_page[pi] = [r]
                    boxed_any = True
                    break
        
        # If all occurrences overlap quotes, don't box any metadata for this field
        if not boxed_any:
            return

        # Place the annotation (callout) on page 1
        # For placement heuristics, we use the union of page-1 targets if any, else union of first found page.
        if 0 in cleaned_targets_by_page:
            placement_targets = cleaned_targets_by_page[0]
        else:
            first_pi = sorted(cleaned_targets_by_page.keys())[0]
            placement_targets = cleaned_targets_by_page[first_pi]

        callout_rect, wrapped_text, fs, _safe = _place_annotation_in_margin(
            page1, placement_targets, occupied_callouts, label,
            left_annotation_count, right_annotation_count
        )

        footer_no_go = fitz.Rect(NO_GO_RECT) & page1.rect
        if footer_no_go.width > 0 and footer_no_go.height > 0 and callout_rect.intersects(footer_no_go):
            shift = (callout_rect.y1 - footer_no_go.y0) + EDGE_PAD
            callout_rect = _shift_rect_up(callout_rect, shift, min_y=EDGE_PAD)

        callout_rect = _ensure_min_size(callout_rect, page1.rect)
        if not _rect_is_valid(callout_rect):
            return

        # White backing + text
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        final_rect, _ret, _final_fs = _insert_textbox_fit(
            page1,
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )

        occupied_callouts.append(final_rect)
        
        # Track which side this annotation is on
        if final_rect.x0 < page1.rect.width / 2:
            left_annotation_count += 1
        else:
            right_annotation_count += 1

        # Store connector instructions to draw after all callouts exist
        connectors_to_draw.append(
            {
                "final_rect": final_rect,
                "connect_policy": connect_policy,
                "targets_by_page": cleaned_targets_by_page,
            }
        )

    # --- Criterion-specific annotation (for first quote term) ---
    # Determine the criterion-specific label based on criterion_id
    criterion_label = None
    performance_date_str = meta.get("performance_date")

    criterion_str = str(criterion_id or "").lower().strip()
    criterion_category = "past" if "past" in criterion_str else "future" if "future" in criterion_str else None
    criterion_match = re.search(r"\d+", criterion_str)
    criterion_num = int(criterion_match.group(0)) if criterion_match else None

    if criterion_num == 1:
        criterion_label = "Beneficiary receives award."
    elif criterion_num == 3:
        summary = _summarize_positive_description(first_quote_term or "")
        summary_text = summary or "exceptional"
        criterion_label = f"Beneficiary's performance described as {summary_text}"
    elif criterion_num == 5:
        criterion_label = "Beneficiary's successes are critically acclaimed"
    elif criterion_num == 6:
        criterion_label = "Beneficiary's achievements received recognition from industry experts"
    # Criteria 2, 4, 7 get no quote-term annotation
    
    # Apply criterion-specific annotation to first quote term if we have one
    if criterion_label and quote_targets_by_term:
        # Connect only the first quote occurrence per page (across all terms)
        annotated_targets_by_page: Dict[int, List[fitz.Rect]] = {}
        for pi, rects in quote_hits_by_page.items():
            rects = _dedupe_rects(rects, pad=1.0)
            rects = sorted(rects, key=lambda r: (r.y0, r.x0))
            if rects:
                annotated_targets_by_page[pi] = rects

        if annotated_targets_by_page:
            # Place the criterion annotation
            if 0 in annotated_targets_by_page:
                placement_targets = annotated_targets_by_page[0]
            else:
                first_pi = sorted(annotated_targets_by_page.keys())[0]
                placement_targets = annotated_targets_by_page[first_pi]
            
            callout_rect, wrapped_text, fs, _safe = _place_annotation_in_margin(
                page1, placement_targets, occupied_callouts, criterion_label,
                left_annotation_count, right_annotation_count
            )
            
            footer_no_go = fitz.Rect(NO_GO_RECT) & page1.rect
            if footer_no_go.width > 0 and footer_no_go.height > 0 and callout_rect.intersects(footer_no_go):
                shift = (callout_rect.y1 - footer_no_go.y0) + EDGE_PAD
                callout_rect = _shift_rect_up(callout_rect, shift, min_y=EDGE_PAD)
            
            callout_rect = _ensure_min_size(callout_rect, page1.rect)
            
            if _rect_is_valid(callout_rect):
                # White backing + text
                page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
                
                final_rect, _ret, _final_fs = _insert_textbox_fit(
                    page1,
                    callout_rect,
                    wrapped_text,
                    fontname=FONTNAME,
                    fontsize=fs,
                    color=RED,
                    align=fitz.TEXT_ALIGN_LEFT,
                    overlay=True,
                )
                
                occupied_callouts.append(final_rect)
                
                # Track which side this annotation is on
                if final_rect.x0 < page1.rect.width / 2:
                    left_annotation_count += 1
                else:
                    right_annotation_count += 1
                
                # Store connector instructions
                connectors_to_draw.append(
                    {
                        "final_rect": final_rect,
                        "connect_policy": "page_best",
                        "targets_by_page": annotated_targets_by_page,
                    }
                )

    # --- Meta labels (criterion-specific metadata annotations) ---
    # For source_url, try multiple variants (with/without protocol, with/without www)
    source_url = meta.get("source_url")
    source_url_variants = []
    if source_url:
        # Try without https://
        without_protocol = source_url.replace('https://', '').replace('http://', '')
        source_url_variants.append(without_protocol)
        
        # Try with just domain (theatermania.com)
        if without_protocol.startswith('www.'):
            without_www = without_protocol.replace('www.', '', 1)
            source_url_variants.append(without_www)
    
    if criterion_num in {1, 2, 3, 4, 5, 6, 7}:
        _do_job(
            "Original source of publication.",
            source_url,
            connect_policy="all",
            also_try_variants=source_url_variants,
        )

    if criterion_num == 1:
        _do_job("Award issuer.", meta.get("venue_name"), connect_policy="all")

    if criterion_num == 4:
        _do_job("Distinguished organization.", meta.get("venue_name"), connect_policy="all")
        _do_job("Distinguished organization.", meta.get("ensemble_name"), connect_policy="all")

    if criterion_num in {2, 4}:
        performance_date_str = meta.get("performance_date")
        if performance_date_str:
            if criterion_category == "past":
                date_label = "past performance"
            elif criterion_category == "future":
                date_label = "future performance"
            else:
                effective_current_date = current_date if current_date is not None else datetime.now()
                perf_date = parse_date(performance_date_str)
                if perf_date and perf_date.date() < effective_current_date.date():
                    date_label = "past performance"
                elif perf_date and perf_date.date() > effective_current_date.date():
                    date_label = "future performance"
                else:
                    date_label = None
            if date_label:
                _do_job(date_label, performance_date_str, connect_policy="all")

    # Second pass: draw connectors AFTER all callouts exist
    for item in connectors_to_draw:
        final_rect = item["final_rect"]
        targets_by_page = item["targets_by_page"]
        connect_policy = item["connect_policy"]
        last_target_page_idx = max(targets_by_page.keys()) if targets_by_page else 0

        # Draw connectors to ALL targets across pages, routed down margins if needed.
        # NOTE: callout is always on page 0.
        callout_page_index = 0

        if connect_policy in {"all", "page_first"}:
            targets_by_page = _select_targets_for_connectors(
                targets_by_page,
                policy=connect_policy,
            )

        for pi, rects in targets_by_page.items():
            if connect_policy == "single" and rects:
                rects = rects[:1]

            if connect_policy == "page_best":
                if pi == 0:
                    red_boxes = quote_hits_by_page.get(0, [])
                    best = _choose_best_target_on_page(
                        page=page1,
                        candidates=rects,
                        margin_x=EDGE_PAD,
                        callout_left=final_rect.x0 < page1.rect.width / 2,
                        avoid_rects=red_boxes,
                    )
                    if best is not None:
                        s, e = _edge_to_edge_points(final_rect, best)
                        obstacles = quote_hits_by_page.get(0, []) + occupied_callouts
                        obstacles = [
                            o for o in obstacles
                            if not o.intersects(inflate_rect(best, OVERLAP_TOLERANCE))
                        ]
                        obstacles = [o for o in obstacles if not o.intersects(final_rect)]
                        _draw_routed_line(page1, s, e, obstacles)
                else:
                    target_page = doc.load_page(pi)
                    callout_page = doc.load_page(callout_page_index)
                    callout_left, left_margin_x, right_margin_x = _choose_callout_margin_side(
                        final_rect, callout_page.rect
                    )
                    margin_x = left_margin_x if callout_left else right_margin_x
                    red_boxes = quote_hits_by_page.get(pi, [])
                    best = _choose_best_target_on_page(
                        page=target_page,
                        candidates=rects,
                        margin_x=margin_x,
                        callout_left=callout_left,
                        avoid_rects=red_boxes,
                    )
                    if best is not None:
                        _draw_multipage_connector(
                            doc,
                            callout_page_index,
                            final_rect,
                            pi,
                            best,
                            occupied_callouts=occupied_callouts,
                            last_target_page_idx=last_target_page_idx,
                        )
            else:
                for r in rects:
                    if pi == 0:
                        # Same-page: route around obstacles
                        s, e = _edge_to_edge_points(final_rect, r)
                        
                        # Collect obstacles (all red boxes + all OTHER annotations, not this one!)
                        obstacles = quote_hits_by_page.get(0, []) + occupied_callouts
                        obstacles = [
                            o for o in obstacles
                            if not o.intersects(inflate_rect(r, OVERLAP_TOLERANCE))
                        ]
                        obstacles = [o for o in obstacles if not o.intersects(final_rect)]
                        
                        # Use smart routing
                        _draw_routed_line(page1, s, e, obstacles)
                    else:
                        _draw_multipage_connector(
                            doc,
                            callout_page_index,
                            final_rect,
                            pi,
                            r,
                            occupied_callouts=occupied_callouts,
                            last_target_page_idx=last_target_page_idx,
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
