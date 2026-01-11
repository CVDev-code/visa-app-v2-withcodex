import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"
FONT_SIZES = [12, 11, 10, 9, 8]

# ---- spacing knobs ----
EDGE_PAD = 12.0
GAP_FROM_TEXT_BLOCKS = 8.0
GAP_FROM_HIGHLIGHTS = 10.0
GAP_BETWEEN_CALLOUTS = 8.0
ENDPOINT_PULLBACK = 1.5

# For quote search robustness
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18


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


def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect, steps: int = 28) -> bool:
    """Sample points along segment; good enough for scoring line crossings."""
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)):
            return True
    return False


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
    """
    Detect the body text area by analyzing words (robust percentiles).
    Ignores top/bottom ~12% of the page to avoid header/footer.
    """
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

        # 5th and 95th percentiles
        li = int(len(x0s) * 0.05)
        ri = int(len(x1s) * 0.95)

        text_left = x0s[max(0, li)]
        text_right = x1s[min(len(x1s) - 1, ri)]

        # keep at least 8% margins
        text_left = max(pr.width * 0.08, text_left)
        text_right = min(pr.width * 0.92, text_right)

        # sanity
        if text_right <= text_left + 50:
            return _get_fallback_text_area(page)

        return fitz.Rect(text_left, header_limit, text_right, footer_limit)
    except Exception:
        return _get_fallback_text_area(page)


# ============================================================
# Text wrapping (simple + reliable)
# ============================================================

def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    """
    Wrap words to the given width, try font sizes from FONT_SIZES.
    Returns: (fontsize, wrapped_text, width_used, height_estimate)
    """
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    words = text.split()
    max_h = 180.0  # allow taller; actual fit is checked by insert_textbox

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
    """
    Insert text; if it doesn't fit (negative return), expand vertically and/or shrink font.
    Returns (final_rect, ret, final_fontsize).
    """
    pr = page.rect
    r = fitz.Rect(rect)
    fs = int(fontsize)

    def attempt(rr: fitz.Rect, fsize: int) -> float:
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
# NEW: edge-to-edge connector + crossing score
# ============================================================

def _connector_endpoints_edge_to_edge(callout_rect: fitz.Rect, target_rect: fitz.Rect) -> Tuple[fitz.Point, fitz.Point]:
    """
    Start on callout edge facing target; end on TARGET EDGE facing callout.
    This avoids stabbing into the middle of the red box.
    """
    tc = _center(target_rect)
    cc = _center(callout_rect)

    # Start point on callout edge
    cy = cc.y
    if callout_rect.x1 <= target_rect.x0:
        start = fitz.Point(callout_rect.x1, cy)  # right edge of callout
    elif callout_rect.x0 >= target_rect.x1:
        start = fitz.Point(callout_rect.x0, cy)  # left edge of callout
    else:
        # callout horizontally overlaps target; fall back to top/bottom edge decision
        if cc.y < tc.y:
            start = fitz.Point(cc.x, callout_rect.y1)
        else:
            start = fitz.Point(cc.x, callout_rect.y0)

    # End point on target edge (closest edge to callout)
    # Clamp y to target vertical span for nicer "edge touch"
    y_on_target = min(max(cy, target_rect.y0 + 1.0), target_rect.y1 - 1.0)

    if callout_rect.x1 <= target_rect.x0:
        end = fitz.Point(target_rect.x0, y_on_target)  # LEFT edge of target
    elif callout_rect.x0 >= target_rect.x1:
        end = fitz.Point(target_rect.x1, y_on_target)  # RIGHT edge of target
    else:
        # fallback vertical edge
        x_on_target = min(max(cc.x, target_rect.x0 + 1.0), target_rect.x1 - 1.0)
        if cc.y < tc.y:
            end = fitz.Point(x_on_target, target_rect.y0)  # TOP edge
        else:
            end = fitz.Point(x_on_target, target_rect.y1)  # BOTTOM edge

    # Pull back slightly so the line ends just outside the red box, not inside it
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)
    return start, end


def _get_text_obstacles_for_scoring(page: fitz.Page, targets: List[fitz.Rect]) -> List[fitz.Rect]:
    """
    Obstacles = text blocks inflated slightly, excluding the target highlights themselves.
    We score crossings against these.
    """
    obs: List[fitz.Rect] = []
    try:
        blocks = page.get_text("blocks") or []
        for b in blocks:
            r = inflate_rect(fitz.Rect(b[:4]), 1.5)  # small inflation for sensitivity
            obs.append(r)
    except Exception:
        pass

    # Exclude target areas so we don't penalize "touching" the intended region
    target_buf = [inflate_rect(t, 2.0) for t in targets]
    filtered: List[fitz.Rect] = []
    for r in obs:
        if any(r.intersects(tb) for tb in target_buf):
            continue
        filtered.append(r)

    return filtered


def _connector_crossing_score(
    start: fitz.Point,
    end: fitz.Point,
    obstacles: List[fitz.Rect],
) -> int:
    """
    Lower is better. Count how many text obstacles the connector passes through.
    """
    hits = 0
    for r in obstacles:
        if _segment_hits_rect(start, end, r):
            hits += 1
    return hits


def _draw_connector(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect):
    s, e = _connector_endpoints_edge_to_edge(callout_rect, target_rect)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)


# ============================================================
# Margin placement (KEEP same lanes) + choose best side/offset by scoring crossings
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

    # NEW: don't let callout land on top of the highlighted thing (e.g. URL footer target)
    target_no_go = inflate_rect(target_union, GAP_FROM_HIGHLIGHTS)

    MIN_CALLOUT_WIDTH = 55.0
    MAX_CALLOUT_WIDTH = 180.0
    EDGE_BUFFER = 8.0

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
        return fallback, label, 8, False

    # Prefer the side nearer the target (so less line across text)
    page_mid_x = pr.width / 2.0
    target_side_pref = "left" if target_c.x < page_mid_x else "right"
    lanes.sort(key=lambda t: 0 if t[0] == target_side_pref else 1)

    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    # NEW: prefer angled connectors: try small offsets before dy=0
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

            if side == "left":
                x1 = x1_lane - 5.0
                x0 = max(x0_lane, x1 - w_used)
            else:
                x0 = x0_lane + 5.0
                x1 = min(x1_lane, x0 + w_used)

            cand = fitz.Rect(x0, y0, x1, y1)
            if cand.intersects(target_no_go):
                continue

            # Hard rule: stay out of body text area horizontally
            if cand.intersects(text_area):
                continue

            # NEW: never cover the target highlight (prevents URL being covered)
            if cand.intersects(target_no_go):
                continue

            conflicts = any(cand.intersects(o) for o in occupied_buf)
            safe = not conflicts

            # score: prefer safe, prefer nearer side, prefer small non-zero dy (angled)
            dx = abs(_center(cand).x - target_c.x)

            # NEW: tiny penalty for dy==0 so angled is preferred
            dy_zero_penalty = 5.0 if dy == 0 else 0.0

            score = (0 if safe else 10_000) + dx * 0.8 + abs(dy) * 0.15 + dy_zero_penalty

            if best is None or score < best[0]:
                best = (score, cand, wrapped_text, fs, safe)

            if safe and dy != 0:
                # Good enough: safe + angled
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

    if side == "left":
        x1 = x1_lane - 5.0
        x0 = max(x0_lane, x1 - w_used)
    else:
        x0 = x0_lane + 5.0
        x1 = min(x1_lane, x0 + w_used)

    cand = fitz.Rect(x0, y0, x1, y1)
    return cand, wrapped_text, fs, False


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

    # A) Quote highlights (all pages)
    for page in doc:
        for term in (quote_terms or []):
            rects = _search_term(page, term)
            for r in rects:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1

    # B) Metadata callouts (page 1)
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
                targets.extend(page1.search_for(needle))
            except Exception:
                pass

        if not targets:
            return

        for t in targets:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_meta_hits += len(targets)

        callout_rect, wrapped_text, fs, safe = _place_annotation_in_margin(
            page1, targets, occupied_callouts, label
        )

        # Always draw white backing
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        # Fit-guaranteed text insertion
        final_rect, ret, final_fs = _insert_textbox_fit(
            page1,
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )

        # Connector(s) — now edge-to-edge
        target_union = _union_rect(targets)

        if connect_policy == "all":
            for t in targets:
                _draw_connector(page1, final_rect, t)
        elif connect_policy == "single":
            _draw_connector(page1, final_rect, targets[0])
        else:
            _draw_connector(page1, final_rect, target_union)

        occupied_callouts.append(final_rect)

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
