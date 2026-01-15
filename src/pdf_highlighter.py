import io
import math
import re
import calendar
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"
FONT_SIZES = [11, 10, 9, 8]

# ---- footer no-go zone (page coordinates; PyMuPDF = top-left origin) ----
NO_GO_RECT = fitz.Rect(
    21.00,   # left
    816.00,  # top
    411.26,  # right
    830.00   # bottom
)

# ---- spacing knobs ----
EDGE_PAD = 12.0
GAP_FROM_TEXT_BLOCKS = 8.0
GAP_FROM_HIGHLIGHTS = 10.0
GAP_BETWEEN_CALLOUTS = 8.0
ENDPOINT_PULLBACK = 1.5

# Arrowhead
ARROW_LEN = 9.0
ARROW_HALF_WIDTH = 4.5

# For quote search robustness
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18

PAST_CRITERIA = {"2_past", "4_past"}
FUTURE_CRITERIA = {"2_future", "4_future"}

_DATE_INPUT_FORMATS = [
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
]


def _normalize_date_string(value: str) -> str:
    cleaned = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _generate_date_variants(value: Optional[str]) -> List[str]:
    if not value:
        return []

    cleaned = _normalize_date_string(str(value))
    if not cleaned:
        return []

    parsed = None
    for fmt in _DATE_INPUT_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue

    if not parsed:
        return []

    def _strip_leading_zero(formatted: str, sep: str) -> str:
        parts = formatted.split(sep)
        return sep.join(part.lstrip("0") or "0" for part in parts)

    variants = {
        parsed.strftime("%B %d, %Y"),
        parsed.strftime("%b %d, %Y"),
        parsed.strftime("%B %d %Y"),
        parsed.strftime("%b %d %Y"),
        parsed.strftime("%d %B %Y"),
        parsed.strftime("%d %b %Y"),
        parsed.strftime("%m/%d/%Y"),
        parsed.strftime("%d/%m/%Y"),
        parsed.strftime("%m/%d/%y"),
        parsed.strftime("%d/%m/%y"),
        parsed.strftime("%m-%d-%Y"),
        parsed.strftime("%d-%m-%Y"),
        parsed.strftime("%Y-%m-%d"),
    }

    variants.update(
        {
            _strip_leading_zero(parsed.strftime("%m/%d/%Y"), "/"),
            _strip_leading_zero(parsed.strftime("%d/%m/%Y"), "/"),
            _strip_leading_zero(parsed.strftime("%m/%d/%y"), "/"),
            _strip_leading_zero(parsed.strftime("%d/%m/%y"), "/"),
            _strip_leading_zero(parsed.strftime("%m-%d-%Y"), "-"),
            _strip_leading_zero(parsed.strftime("%d-%m-%Y"), "-"),
        }
    )

    return sorted({v.strip() for v in variants if v.strip()})


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
        return 11, "", box_width, 24.0

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
        return r, -1.0, fs

    def attempt(rr: fitz.Rect, fsize: int) -> float:
        rr = _ensure_min_size(rr, pr)
        if not _rect_is_valid(rr):
            return -1.0
        return page.insert_textbox(
            rr,
            text,
            fontname=fontname,
            fontsize=fsize,
            color=color,
            align=align,
            overlay=overlay,
        )

    ret = attempt(r, fs)

    it = 0
    while ret < 0 and it < max_expand_iters:
        need = (-ret) + extra_pad_each_iter
        r.y0 -= need / 2.0
        r.y1 += need / 2.0
        r.y0 = max(2.0, r.y0)
        r.y1 = min(pr.height - 2.0, r.y1)
        ret = attempt(r, fs)
        it += 1

    shrink_tries = 0
    while ret < 0 and fs > FONT_SIZES[-1] and shrink_tries < 4:
        fs -= 1
        r = fitz.Rect(rect)
        ret = attempt(r, fs)

        it = 0
        while ret < 0 and it < max_expand_iters:
            need = (-ret) + extra_pad_each_iter
            r.y0 -= need / 2.0
            r.y1 += need / 2.0
            r.y0 = max(2.0, r.y0)
            r.y1 = min(pr.height - 2.0, r.y1)
            ret = attempt(r, fs)
            it += 1

        shrink_tries += 1

    return r, ret, fs


# ============================================================
# De-duplication for overlapping/nested hits
# ============================================================

def _rect_area(r: fitz.Rect) -> float:
    return max(0.0, (r.x1 - r.x0) * (r.y1 - r.y0))


def _dedupe_rects(rects: List[fitz.Rect], pad: float = 1.0) -> List[fitz.Rect]:
    """
    Remove nested duplicates: if one rect is fully contained in a larger rect, keep only the larger.
    Works well for cases like:
      "Jasurbek", "Khaydarov", "Jasurbek Khaydarov", "Jasurbek Khaydarov’s Polydorus..."
    """
    if not rects:
        return []

    rr = [fitz.Rect(r) for r in rects]
    rr.sort(key=lambda r: _rect_area(r), reverse=True)

    kept: List[fitz.Rect] = []
    for r in rr:
        rbuf = inflate_rect(r, pad)
        contained = False
        for k in kept:
            if inflate_rect(k, pad).contains(rbuf):
                contained = True
                break
        if not contained:
            kept.append(r)

    kept.sort(key=lambda r: (r.y0, r.x0))
    return kept


# ============================================================
# Robust search helpers
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
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
                hits.extend(page.search_for(chunk, flags=flags))
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
# Arrow drawing (tip ends at end-point)
# ============================================================

def _draw_arrowhead(page: fitz.Page, start: fitz.Point, end: fitz.Point):
    """
    Draw a small filled triangle arrowhead with its TIP exactly at `end`.
    """
    vx = end.x - start.x
    vy = end.y - start.y
    d = math.hypot(vx, vy)
    if d == 0:
        return

    ux, uy = vx / d, vy / d  # direction

    # base center is behind tip
    bx = end.x - ux * ARROW_LEN
    by = end.y - uy * ARROW_LEN

    # perpendicular
    px = -uy
    py = ux

    p1 = fitz.Point(bx + px * ARROW_HALF_WIDTH, by + py * ARROW_HALF_WIDTH)
    p2 = fitz.Point(bx - px * ARROW_HALF_WIDTH, by - py * ARROW_HALF_WIDTH)
    tip = fitz.Point(end.x, end.y)

    page.draw_polyline([p1, tip, p2, p1], color=RED, fill=RED, width=0.0)


def _draw_line(page: fitz.Page, a: fitz.Point, b: fitz.Point):
    page.draw_line(a, b, color=RED, width=LINE_WIDTH)


# ============================================================
# Connector endpoints (edge-to-edge with pullback)
# ============================================================

def _edge_to_edge_points(callout_rect: fitz.Rect, target_rect: fitz.Rect) -> Tuple[fitz.Point, fitz.Point]:
    tc = _center(target_rect)
    cc = _center(callout_rect)

    cy = cc.y
    if callout_rect.x1 <= target_rect.x0:
        start = fitz.Point(callout_rect.x1, cy)
    elif callout_rect.x0 >= target_rect.x1:
        start = fitz.Point(callout_rect.x0, cy)
    else:
        if cc.y < tc.y:
            start = fitz.Point(cc.x, callout_rect.y1)
        else:
            start = fitz.Point(cc.x, callout_rect.y0)

    y_on_target = min(max(cy, target_rect.y0 + 1.0), target_rect.y1 - 1.0)

    if callout_rect.x1 <= target_rect.x0:
        end = fitz.Point(target_rect.x0, y_on_target)
    elif callout_rect.x0 >= target_rect.x1:
        end = fitz.Point(target_rect.x1, y_on_target)
    else:
        x_on_target = min(max(cc.x, target_rect.x0 + 1.0), target_rect.x1 - 1.0)
        if cc.y < tc.y:
            end = fitz.Point(x_on_target, target_rect.y0)
        else:
            end = fitz.Point(x_on_target, target_rect.y1)

    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)
    return start, end


# ============================================================
# Multi-page routing: down the page margin(s) until target page
# ============================================================

def _draw_multipage_connector(
    doc: fitz.Document,
    callout_page_index: int,
    callout_rect: fitz.Rect,
    target_page_index: int,
    target_rect: fitz.Rect,
):
    """
    Draw:
      - Page 1 (annotation page): from callout edge -> gutter point, then down to bottom
      - Intermediate pages: vertical gutter line top->bottom
      - Target page: gutter line top->target_y, then across to target edge, with arrowhead at end
    """
    callout_page = doc.load_page(callout_page_index)
    pr = callout_page.rect

    # Choose gutter based on which side callout is on
    callout_c = _center(callout_rect)
    gutter_side = "right" if callout_c.x >= pr.width / 2 else "left"
    gutter_x = pr.width - EDGE_PAD if gutter_side == "right" else EDGE_PAD

    # Page 1: from callout to gutter at same y, then down
    union = target_rect
    s, _ = _edge_to_edge_points(callout_rect, union)

    y_start = min(max(s.y, EDGE_PAD), pr.height - EDGE_PAD)
    p_gutter_start = fitz.Point(gutter_x, y_start)
    p_gutter_bottom = fitz.Point(gutter_x, pr.height - EDGE_PAD)

    _draw_line(callout_page, s, p_gutter_start)
    _draw_line(callout_page, p_gutter_start, p_gutter_bottom)

    # Intermediate pages
    for pi in range(callout_page_index + 1, target_page_index):
        p = doc.load_page(pi)
        pr_i = p.rect
        gx = pr_i.width - EDGE_PAD if gutter_side == "right" else EDGE_PAD
        _draw_line(p, fitz.Point(gx, EDGE_PAD), fitz.Point(gx, pr_i.height - EDGE_PAD))

    # Target page: gutter top -> target_y, then across to target edge with arrowhead
    tp = doc.load_page(target_page_index)
    pr_t = tp.rect
    gx_t = pr_t.width - EDGE_PAD if gutter_side == "right" else EDGE_PAD

    tc = _center(target_rect)
    y_target = min(max(tc.y, EDGE_PAD), pr_t.height - EDGE_PAD)

    p_top = fitz.Point(gx_t, EDGE_PAD)
    p_mid = fitz.Point(gx_t, y_target)

    # End point on target edge (pulled back so arrow tip doesn't enter red box)
    # We "pretend" start is at gutter point for pullback direction.
    faux_start = fitz.Point(gx_t, y_target)
    if gutter_side == "right":
        end_raw = fitz.Point(target_rect.x1, min(max(y_target, target_rect.y0 + 1.0), target_rect.y1 - 1.0))
    else:
        end_raw = fitz.Point(target_rect.x0, min(max(y_target, target_rect.y0 + 1.0), target_rect.y1 - 1.0))

    end = _pull_back_point(faux_start, end_raw, ENDPOINT_PULLBACK)

    _draw_line(tp, p_top, p_mid)
    _draw_line(tp, p_mid, end)

    # Arrowhead: tip exactly at end
    _draw_arrowhead(tp, p_mid, end)


# ============================================================
# Margin placement (unchanged behaviour; first-page callouts)
# ============================================================

def _place_annotation_in_margin(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied_callouts: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    text_area = _detect_actual_text_area(page)
    pr = page.rect
    target_union = _union_rect(targets)
    target_c = _center(target_union)
    target_y = target_c.y

    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)
    footer_no_go = fitz.Rect(NO_GO_RECT) & pr

    MIN_CALLOUT_WIDTH = 55.0
    MAX_CALLOUT_WIDTH = 180.0
    EDGE_BUFFER = 8.0
    MIN_H = 12.0

    left_lane = (EDGE_BUFFER, max(EDGE_BUFFER, text_area.x0 - EDGE_BUFFER))
    right_lane = (min(pr.width - EDGE_BUFFER, text_area.x1 + EDGE_BUFFER), pr.width - EDGE_BUFFER)

    lanes = []
    lw = left_lane[1] - left_lane[0]
    rw = right_lane[1] - right_lane[0]
    if lw >= MIN_CALLOUT_WIDTH:
        lanes.append(("left", left_lane[0], left_lane[1], lw))
    if rw >= MIN_CALLOUT_WIDTH:
        lanes.append(("right", right_lane[0], right_lane[1], rw))

    if not lanes:
        fallback = fitz.Rect(
            EDGE_BUFFER,
            max(EDGE_BUFFER, target_y - 20),
            EDGE_BUFFER + 120,
            min(pr.height - EDGE_BUFFER, target_y + 20),
        )
        return _ensure_min_size(fallback, pr), label, 8, False

    page_mid_x = pr.width / 2.0
    target_side_pref = "left" if target_c.x < page_mid_x else "right"
    lanes.sort(key=lambda t: 0 if t[0] == target_side_pref else 1)

    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]
    scan = [12, -12, 24, -24, 36, -36, 48, -48, 60, -60, 72, -72, 0]

    best = None  # (score, rect, wrapped, fs, safe)

    for side, x0_lane, x1_lane, lane_w in lanes:
        usable_w = min(MAX_CALLOUT_WIDTH, lane_w)
        fs, wrapped_text, w_used, h_needed = _optimize_layout_for_margin(label, usable_w)
        w_used = min(w_used, usable_w)
        if w_used < MIN_CALLOUT_WIDTH:
            continue

        for dy in scan:
            y0 = target_y + dy - h_needed / 2.0
            y1 = target_y + dy + h_needed / 2.0

            y0 = max(EDGE_BUFFER, y0)
            y1 = min(pr.height - EDGE_BUFFER, y1)

            if (y1 - y0) < MIN_H:
                y1 = min(pr.height - EDGE_BUFFER, y0 + MIN_H)
                if (y1 - y0) < MIN_H:
                    y0 = max(EDGE_BUFFER, y1 - MIN_H)

            if side == "left":
                x1 = x1_lane - 5.0
                x0 = max(x0_lane, x1 - w_used)
            else:
                x0 = x0_lane + 5.0
                x1 = min(x1_lane, x0 + w_used)

            cand = fitz.Rect(x0, y0, x1, y1)
            cand = _ensure_min_size(cand, pr)

            if cand.intersects(target_no_go):
                continue
            if cand.intersects(text_area):
                continue
            if footer_no_go.width > 0 and footer_no_go.height > 0 and cand.intersects(footer_no_go):
                continue

            conflicts = any(cand.intersects(o) for o in occupied_buf)
            safe = not conflicts

            dx = abs(_center(cand).x - target_c.x)
            dy_zero_penalty = 5.0 if dy == 0 else 0.0
            score = (0 if safe else 10_000) + dx * 0.8 + abs(dy) * 0.15 + dy_zero_penalty

            if best is None or score < best[0]:
                best = (score, cand, wrapped_text, fs, safe)

            if safe and dy != 0:
                return cand, wrapped_text, fs, True

    if best:
        _, cand, wrapped_text, fs, safe = best
        return cand, wrapped_text, fs, safe

    # last resort fallback
    side, x0_lane, x1_lane, lane_w = lanes[0]
    usable_w = min(MAX_CALLOUT_WIDTH, lane_w)
    fs, wrapped_text, w_used, h_needed = _optimize_layout_for_margin(label, usable_w)
    w_used = min(w_used, usable_w)

    y0 = max(EDGE_BUFFER, target_y - h_needed / 2.0)
    y1 = min(pr.height - EDGE_BUFFER, target_y + h_needed / 2.0)
    if (y1 - y0) < MIN_H:
        y1 = min(pr.height - EDGE_BUFFER, y0 + MIN_H)

    if side == "left":
        x1 = x1_lane - 5.0
        x0 = max(x0_lane, x1 - w_used)
    else:
        x0 = x0_lane + 5.0
        x1 = min(x1_lane, x0 + w_used)

    cand = fitz.Rect(x0, y0, x1, y1)

    if footer_no_go.width > 0 and footer_no_go.height > 0 and cand.intersects(footer_no_go):
        shift = (cand.y1 - footer_no_go.y0) + EDGE_BUFFER
        cand = _shift_rect_up(cand, shift, min_y=EDGE_BUFFER)

    cand = _ensure_min_size(cand, pr)
    return cand, wrapped_text, fs, False


# ============================================================
# Main annotation entrypoint
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

    # Track quote hits with page index for multi-page connectors
    quote_hits_by_page: Dict[int, List[fitz.Rect]] = {}

    # A) Quote highlights (all pages) + dedupe per page
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_hits: List[fitz.Rect] = []

        for term in (quote_terms or []):
            rects = _search_term(page, term)
            page_hits.extend(rects)

        page_hits = _dedupe_rects(page_hits, pad=1.0)
        if page_hits:
            quote_hits_by_page[page_index] = page_hits

        for r in page_hits:
            page.draw_rect(r, color=RED, width=BOX_WIDTH)
            total_quote_hits += 1

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

        # Draw red boxes on all pages where found
        for pi, rects in cleaned_targets_by_page.items():
            p = doc.load_page(pi)
            for r in rects:
                p.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_meta_hits += 1

        # Place the annotation (callout) on page 1
        # For placement heuristics, we use the union of page-1 targets if any, else union of first found page.
        if 0 in cleaned_targets_by_page:
            placement_targets = cleaned_targets_by_page[0]
        else:
            first_pi = sorted(cleaned_targets_by_page.keys())[0]
            placement_targets = cleaned_targets_by_page[first_pi]

        callout_rect, wrapped_text, fs, _safe = _place_annotation_in_margin(
            page1, placement_targets, occupied_callouts, label
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

        # Store connector instructions to draw after all callouts exist
        connectors_to_draw.append(
            {
                "final_rect": final_rect,
                "connect_policy": connect_policy,
                "targets_by_page": cleaned_targets_by_page,
            }
        )

    # --- Meta labels (now includes ensemble) ---
    _do_job("Original source of publication.", meta.get("source_url"), connect_policy="all")
    _do_job("Venue is a distinguished organization.", meta.get("venue_name"), connect_policy="all")
    _do_job("Ensemble is a distinguished organization.", meta.get("ensemble_name"), connect_policy="all")
    
    if criterion_id in PAST_CRITERIA:
        performance_label = "Past performance date."
    elif criterion_id in FUTURE_CRITERIA:
        performance_label = "Future performance date."
    else:
        performance_label = "Performance date."

    performance_date_variants = _generate_date_variants(meta.get("performance_date"))
    _do_job(
        performance_label,
        meta.get("performance_date"),
        connect_policy="all",
        also_try_variants=performance_date_variants,
    )

    # Beneficiary targets (still value-driven)
    _do_job(
        "Beneficiary named as a lead role.",
        meta.get("beneficiary_name"),
        connect_policy="all",
        also_try_variants=meta.get("beneficiary_variants") or [],
    )

    # Second pass: draw connectors AFTER all callouts exist
    for item in connectors_to_draw:
        final_rect = item["final_rect"]
        targets_by_page = item["targets_by_page"]
        connect_policy = item["connect_policy"]

        # Draw connectors to ALL targets across pages, routed down margins if needed.
        # NOTE: callout is always on page 0.
        callout_page_index = 0

        for pi, rects in targets_by_page.items():
            if connect_policy == "single" and rects:
                rects = rects[:1]

            for r in rects:
                if pi == 0:
                    # Same-page: simple line + arrowhead
                    s, e = _edge_to_edge_points(final_rect, r)
                    page1.draw_line(s, e, color=RED, width=LINE_WIDTH)
                    _draw_arrowhead(page1, s, e)
                else:
                    _draw_multipage_connector(doc, callout_page_index, final_rect, pi, r)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
