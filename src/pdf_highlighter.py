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
FONT_SIZES = [12, 11, 10]

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
# Page blockers (for callout placement)
# ============================================================

def _page_blockers(page: fitz.Page, pad: float = GAP_FROM_TEXT_BLOCKS) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []

    # Text blocks
    try:
        for b in page.get_text("blocks"):
            blockers.append(inflate_rect(fitz.Rect(b[:4]), pad))
    except Exception:
        pass

    # Images
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                for r in page.get_image_rects(xref):
                    blockers.append(inflate_rect(fitz.Rect(r), pad))
            except Exception:
                pass
    except Exception:
        pass

    # Vector drawings
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r:
                blockers.append(inflate_rect(fitz.Rect(r), pad))
    except Exception:
        pass

    return blockers


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)


# ============================================================
# Callout text wrapping (prefers 12, then 11, then 10)
# ============================================================

def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    """
    Returns (fontsize, wrapped_text, width, height).
    Wraps to fit box_width; if box_width is very narrow, falls back to 1 word per line.
    """
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    words = text.split()
    usable_w = max(10.0, box_width - 10.0)

    # If margin is tiny, force 1 word per line (your extreme stacking case)
    if usable_w < 55.0 and len(words) > 1:
        fs = 11 if 11 in FONT_SIZES else FONT_SIZES[-1]
        lines = words[:]  # one per line
        h = (len(lines) * fs * 1.22) + 10.0
        return fs, "\n".join(lines), box_width, h

    for fs in FONT_SIZES:
        lines: List[str] = []
        cur: List[str] = []

        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    cur = [w]
                else:
                    # single word doesn't fit; still place it on its own line
                    lines.append(w)
                    cur = []
        if cur:
            lines.append(" ".join(cur))

        h = (len(lines) * fs * 1.22) + 10.0
        # accept if reasonably compact
        if h <= 110.0 or fs == FONT_SIZES[-1]:
            return fs, "\n".join(lines), box_width, h

    return FONT_SIZES[-1], text, box_width, 44.0


# ============================================================
# FIXED: margin placement with (A) conditional scan, (B) skip OOB,
# and (C) blocker filtering to the margin lane.
# ============================================================

def _choose_best_margin_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2

    DEFAULT_W = 130.0
    MIN_W = 45.0  # allow very narrow; wrapping will stack

    # Build content bbox from OCR/text blocks (the area we refuse to cover)
    try:
        blocks = page.get_text("blocks")
        text_rects = [fitz.Rect(b[:4]) for b in blocks]
    except Exception:
        text_rects = []

    content_bbox = inflate_rect(_union_rect(text_rects), GAP_FROM_TEXT_BLOCKS)

    # Keep away from highlights and other callouts
    protected = [content_bbox] + [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]

    # Compute available margin widths *outside* content bbox
    left_avail = max(0.0, content_bbox.x0 - EDGE_PAD)
    right_avail = max(0.0, (pr.width - EDGE_PAD) - content_bbox.x1)

    # Decide side; prefer the side with more space
    side_order = []
    if right_avail >= left_avail:
        side_order = ["right", "left"]
    else:
        side_order = ["left", "right"]

    def clamp_y(y0: float, y1: float) -> Tuple[float, float]:
        # not a search: just keep it on the page
        if y0 < EDGE_PAD:
            y1 += (EDGE_PAD - y0)
            y0 = EDGE_PAD
        if y1 > pr.height - EDGE_PAD:
            y0 -= (y1 - (pr.height - EDGE_PAD))
            y1 = pr.height - EDGE_PAD
        return y0, y1

    def safe(cand: fitz.Rect) -> bool:
        # Safe = does not cover OCR/body text + does not overlap other callouts
        # (We do NOT care about images/drawings; white background can cover them.)
        return (not _intersects_any(cand, protected)) and (not _intersects_any(cand, occupied_buf))

    best = None  # (score, cand, wrapped, fs, safe)

    for side in side_order:
        if side == "left":
            avail = left_avail
            if avail < MIN_W:
                continue
            w = min(DEFAULT_W, avail)
            x0 = EDGE_PAD
        else:
            avail = right_avail
            if avail < MIN_W:
                continue
            w = min(DEFAULT_W, avail)
            x0 = pr.width - EDGE_PAD - w

        fs, wrapped, w_used, h = _optimize_layout_for_margin(label, w)
        y0 = target_y - h / 2
        y1 = target_y + h / 2
        y0, y1 = clamp_y(y0, y1)

        cand = fitz.Rect(x0, y0, x0 + w_used, y1)

        s = safe(cand)
        # score: prefer safe, then prefer closer-to-ideal y (we're not moving y though),
        # then prefer wider boxes (more readable)
        score = (0 if s else 1_000_000) - (w_used * 0.01)

        if best is None or score < best[0]:
            best = (score, cand, wrapped, fs, s)

        # If we found a safe placement, take it (no need to try the other side)
        if s:
            break

    # If neither margin is viable, fall back to left but mark unsafe (no white background)
    if best is None:
        fs, wrapped, w_used, h = _optimize_layout_for_margin(label, DEFAULT_W)
        y0, y1 = clamp_y(target_y - h / 2, target_y + h / 2)
        cand = fitz.Rect(EDGE_PAD, y0, EDGE_PAD + w_used, y1)
        return cand, wrapped, fs, False

    _, cand, wrapped, fs, is_safe = best
    return cand, wrapped, fs, is_safe


    # ============================================================
    # A) Normal case: place in left/right margin with dynamic width
    # ============================================================
    if not use_top_bottom:
        for side_name, x_start, avail_w in sides:
            # Choose the *actual* width we can fit in this margin
            margin_w = min(DEFAULT_MARGIN_W, avail_w)

            fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w)

            best_for_side = None

            # First try centered
            y0 = target_y - h / 2
            y1 = target_y + h / 2
            if in_bounds(y0, y1):
                cand0 = fitz.Rect(x_start, y0, x_start + w, y1)
                sc, dist = score_candidate(cand0, target_y)
                best_for_side = (sc, cand0, wrapped, fs, is_safe(cand0))
                if best_for_side[-1]:
                    candidates.append(best_for_side)
                    continue  # already safe at dy=0

            # Otherwise scan vertically (skip out-of-bounds; no clamping)
            for dy in search_steps[1:]:
                yc = target_y + dy
                y0 = yc - h / 2
                y1 = yc + h / 2
                if not in_bounds(y0, y1):
                    continue
                cand = fitz.Rect(x_start, y0, x_start + w, y1)
                sc, dist = score_candidate(cand, target_y)
                if best_for_side is None or sc < best_for_side[0]:
                    best_for_side = (sc, cand, wrapped, fs, is_safe(cand))
                if is_safe(cand) and dist < 5.0:
                    break

            if best_for_side:
                candidates.append(best_for_side)

    # ============================================================
    # B) Fallback: top/bottom margin bands (still avoid content bbox)
    # ============================================================
    else:
        # We’ll place the callout in the top band if possible, else bottom band.
        # Use a conservative width so it fits somewhere even on narrow pages.
        # It will wrap to multiple lines as needed.
        band_w = min(DEFAULT_MARGIN_W, pr.width - 2 * EDGE_PAD)
        fs, wrapped, w, h = _optimize_layout_for_margin(label, band_w)

        # Top band candidate (above content)
        top_y0 = EDGE_PAD
        top_y1 = top_y0 + h
        if top_y1 <= content_bbox.y0 - 2.0:
            cand_top = fitz.Rect(EDGE_PAD, top_y0, EDGE_PAD + w, top_y1)
            candidates.append((0.0 if is_safe(cand_top) else 1_000_000, cand_top, wrapped, fs, is_safe(cand_top)))

        # Bottom band candidate (below content)
        bot_y1 = pr.height - EDGE_PAD
        bot_y0 = bot_y1 - h
        if bot_y0 >= content_bbox.y1 + 2.0:
            cand_bot = fitz.Rect(EDGE_PAD, bot_y0, EDGE_PAD + w, bot_y1)
            candidates.append((0.0 if is_safe(cand_bot) else 1_000_000, cand_bot, wrapped, fs, is_safe(cand_bot)))

    # Final selection
    if not candidates:
        # Absolute last resort: place left at EDGE_PAD and mark unsafe (no white box)
        fs, wrapped, w, h = _optimize_layout_for_margin(label, DEFAULT_MARGIN_W)
        cand = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h)
        return cand, wrapped, fs, False

    candidates.sort(key=lambda x: x[0])
    score, cand, wrapped, fs, safe = candidates[0]

    # If it still isn't safe, we explicitly mark unsafe so caller won't paint background.
    if score >= 1_000_000:
        safe = False

    return cand, wrapped, fs, safe


# ============================================================
# Search helpers (robust quote matching)
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

        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(page1, targets, occupied_callouts, label)

        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
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
