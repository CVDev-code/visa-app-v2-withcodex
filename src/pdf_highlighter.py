diff --git a/src/pdf_highlighter.py b/src/pdf_highlighter.py
index 65d65f36492dd9e6ce457878cb4ba27137e981f5..c6a8b762e91b798c13bfc712ec91d96660abcf69 100644
--- a/src/pdf_highlighter.py
+++ b/src/pdf_highlighter.py
@@ -1,47 +1,47 @@
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
 FONT_SIZES = [12, 11, 10]
 
 # ---- spacing knobs ----
 EDGE_PAD = 18.0
 GAP_FROM_TEXT_BLOCKS = 10.0         # keep callout away from original text
 GAP_FROM_HIGHLIGHTS = 14.0
 GAP_BETWEEN_CALLOUTS = 10.0
-ENDPOINT_PULLBACK = 1.5             # line stops at box edge (pulled back)
+ENDPOINT_PULLBACK = 0.0             # line stops at box edge
 
 # ---- quote search robustness ----
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
@@ -54,159 +54,199 @@ def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
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
 
-def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
-    """Edge points to sample endpoints on the target box."""
+def _edge_candidates_toward(rect: fitz.Rect, toward: fitz.Point) -> List[fitz.Point]:
+    """Edge points to sample endpoints on the target box, biased toward caller."""
     cx = rect.x0 + rect.width / 2.0
     cy = rect.y0 + rect.height / 2.0
-    return [
-        fitz.Point(rect.x0, cy),
-        fitz.Point(rect.x1, cy),
-        fitz.Point(cx, rect.y0),
-        fitz.Point(cx, rect.y1),
-        fitz.Point(rect.x0, rect.y0),
-        fitz.Point(rect.x1, rect.y0),
-        fitz.Point(rect.x0, rect.y1),
-        fitz.Point(rect.x1, rect.y1),
-    ]
+
+    dx = toward.x - cx
+    dy = toward.y - cy
+    if abs(dx) >= abs(dy):
+        if dx >= 0:
+            return [fitz.Point(rect.x1, cy), fitz.Point(rect.x1, rect.y0), fitz.Point(rect.x1, rect.y1)]
+        return [fitz.Point(rect.x0, cy), fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x0, rect.y1)]
+    if dy >= 0:
+        return [fitz.Point(cx, rect.y1), fitz.Point(rect.x0, rect.y1), fitz.Point(rect.x1, rect.y1)]
+    return [fitz.Point(cx, rect.y0), fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x1, rect.y0)]
 
 def _straight_connector_best_pair(
     callout_rect: fitz.Rect,
     target_rect: fitz.Rect,
     obstacles: List[fitz.Rect],
 ) -> Tuple[fitz.Point, fitz.Point]:
     """Always returns a straight segment (start,end) minimizing crossings then length."""
     target_center = _center(target_rect)
 
     base = _mid_edge_anchor(callout_rect, target_center)
     nudges = [-14.0, -7.0, 0.0, 7.0, 14.0]
     starts: List[fitz.Point] = []
     for dy in nudges:
         y = min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)
         starts.append(fitz.Point(base.x, y))
 
-    ends = _edge_candidates(target_rect)
+    ends = _edge_candidates_toward(target_rect, base)
 
     best_hits = 10**9
     best_len = 10**9
     best_s, best_e = starts[0], ends[0]
 
     for s in starts:
         for e in ends:
             hits = 0
             for ob in obstacles:
                 if _segment_hits_rect(s, e, ob):
                     hits += 1
             length = math.hypot(e.x - s.x, e.y - s.y)
             if hits < best_hits or (hits == best_hits and length < best_len):
                 best_hits, best_len = hits, length
                 best_s, best_e = s, e
 
     best_e = _pull_back_point(best_s, best_e, ENDPOINT_PULLBACK)
     return best_s, best_e
 
 def _draw_straight_connector(
     page: fitz.Page,
     callout_rect: fitz.Rect,
     target_rect: fitz.Rect,
     obstacles: List[fitz.Rect],
 ):
     s, e = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
     page.draw_line(s, e, color=RED, width=LINE_WIDTH)
 
 # ============================================================
 # Page blockers (for avoiding overlaps) + TEXT ENVELOPE
 # ============================================================
 
-def _page_text_blocks(page: fitz.Page) -> List[fitz.Rect]:
+def _page_text_line_rects(page: fitz.Page) -> List[fitz.Rect]:
+    line_rects: List[fitz.Rect] = []
+    try:
+        words = page.get_text("words")
+        line_map: Dict[Tuple[int, int], fitz.Rect] = {}
+        for w in words:
+            if len(w) < 7:
+                continue
+            rect = fitz.Rect(w[:4])
+            key = (int(w[5]), int(w[6]))
+            if key in line_map:
+                line_map[key] |= rect
+            else:
+                line_map[key] = fitz.Rect(rect)
+        line_rects.extend(line_map.values())
+    except Exception:
+        pass
+    return line_rects
+
+def _page_text_rects(page: fitz.Page) -> List[fitz.Rect]:
+    return _page_text_block_rects(page) + _page_text_line_rects(page)
+
+def _page_text_block_rects(page: fitz.Page) -> List[fitz.Rect]:
     blocks: List[fitz.Rect] = []
     try:
         for b in page.get_text("blocks"):
             blocks.append(fitz.Rect(b[:4]))
     except Exception:
         pass
     return blocks
 
 def _page_images(page: fitz.Page) -> List[fitz.Rect]:
     imgs: List[fitz.Rect] = []
     try:
         for img in page.get_images(full=True):
             xref = img[0]
             for r in page.get_image_rects(xref):
                 imgs.append(fitz.Rect(r))
     except Exception:
         pass
     return imgs
 
 def _page_blockers(page: fitz.Page, pad: float = GAP_FROM_TEXT_BLOCKS) -> List[fitz.Rect]:
     blockers: List[fitz.Rect] = []
-    for r in _page_text_blocks(page):
+    for r in _page_text_rects(page):
         blockers.append(inflate_rect(r, pad))
     for r in _page_images(page):
         blockers.append(inflate_rect(r, pad))
     try:
         for d in page.get_drawings():
             rr = d.get("rect")
             if rr:
                 blockers.append(inflate_rect(fitz.Rect(rr), pad))
     except Exception:
         pass
     return blockers
 
 def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
     """
     A conservative rectangle covering all text on the page.
     We will ONLY place callouts outside this envelope (margins/top/bottom).
     """
-    blocks = _page_text_blocks(page)
+    blocks = _page_text_block_rects(page)
     if not blocks:
         return None
     env = _union_rect(blocks)
     return env
 
+def _line_span_envelope(page: fitz.Page, y0: float, y1: float) -> Optional[fitz.Rect]:
+    lines = [r for r in _page_text_line_rects(page) if r.y1 >= y0 and r.y0 <= y1]
+    if not lines:
+        return None
+    return _union_rect(lines)
+
+def _span_envelope(page: fitz.Page, y0: float, y1: float) -> Optional[fitz.Rect]:
+    spans = [r for r in _page_text_block_rects(page) if r.y1 >= y0 and r.y0 <= y1]
+    lines = _line_span_envelope(page, y0, y1)
+    if lines:
+        spans.append(lines)
+    if not spans:
+        return None
+    span = _union_rect(spans)
+    span.x0 -= GAP_FROM_TEXT_BLOCKS
+    span.x1 += GAP_FROM_TEXT_BLOCKS
+    return span
+
 def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
     return any(r.intersects(o) for o in others)
 
 # ============================================================
 # Callout text wrapping (prefers 12, then 11, then 10)
 # ============================================================
 
 def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
     text = (text or "").strip()
     if not text:
         return 12, "", box_width, 24.0
 
     for fs in FONT_SIZES:
         words = text.split()
         lines: List[str] = []
         cur: List[str] = []
         usable_w = max(10.0, box_width - 10.0)
 
         for w in words:
             trial = " ".join(cur + [w])
             if fitz.get_text_length(trial, fontname=FONTNAME, fontsize=fs) <= usable_w:
                 cur.append(w)
             else:
                 if cur:
                     lines.append(" ".join(cur))
@@ -270,123 +310,162 @@ def _allowed_zones(page: fitz.Page) -> List[fitz.Rect]:
 
     return zones
 
 def _choose_best_spot_margins_only(
     page: fitz.Page,
     targets: List[fitz.Rect],
     occupied: List[fitz.Rect],
     label: str,
 ) -> Tuple[fitz.Rect, str, int, bool]:
     pr = page.rect
     target_union = _union_rect(targets)
     target_center = _center(target_union)
     target_y = target_center.y
 
     blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)
     for t in targets:
         blockers.append(inflate_rect(t, GAP_FROM_HIGHLIGHTS))
     occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]
 
     zones = _allowed_zones(page)
 
     # candidate widths differ: left/right margins are narrow; top/bottom can be wider
     def zone_box_width(z: fitz.Rect) -> float:
         return min(160.0, max(110.0, z.width - 8.0))
 
-    # We’ll try a few Y positions around target_y and pick best (safe, then connector cost)
-    y_offsets = [-70.0, -40.0, -20.0, 0.0, 20.0, 40.0, 70.0]
+    def _candidate_ys(z: fitz.Rect, h: float) -> List[float]:
+        y_offsets = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
+        ys = [target_y + dy for dy in y_offsets]
+        min_cy = z.y0 + h / 2.0
+        max_cy = z.y1 - h / 2.0
+        step = max(h + GAP_BETWEEN_CALLOUTS, 18.0)
+        cy = min_cy
+        while cy <= max_cy + 0.01:
+            ys.append(cy)
+            cy += step
+        unique = sorted(set(ys), key=lambda y: abs(y - target_y))
+        return [y for y in unique if min_cy <= y <= max_cy]
 
     candidates = []
     for z in zones:
         bw = zone_box_width(z)
         fs, wrapped, w, h = _optimize_layout_for_margin(label, bw)
 
-        for dy in y_offsets:
-            cy = target_y + dy
+        for cy in _candidate_ys(z, h):
             y0 = cy - h / 2
             y1 = cy + h / 2
 
             # clamp within zone
             if y0 < z.y0:
                 y1 += (z.y0 - y0)
                 y0 = z.y0
             if y1 > z.y1:
                 y0 -= (y1 - z.y1)
                 y1 = z.y1
             if y0 < z.y0 or y1 > z.y1:
                 continue
 
-            # choose x: if zone is left margin, place against left edge; if right margin, against right edge; else center-ish
-            if z.x1 <= (_text_envelope(page) or pr).x0:
-                x0 = z.x0
-                x1 = min(z.x1, x0 + w)
-            elif z.x0 >= (_text_envelope(page) or pr).x1:
-                x1 = z.x1
-                x0 = max(z.x0, x1 - w)
+            env = _text_envelope(page) or pr
+            span = _span_envelope(page, y0, y1)
+            span_left = span.x0 if span else env.x0
+            span_right = span.x1 if span else env.x1
+
+            if z.x1 <= env.x0:
+                max_right = min(z.x1, span_left - GAP_FROM_TEXT_BLOCKS)
+                if max_right - z.x0 < 40.0:
+                    continue
+                x_positions = [z.x0]
+                max_x1 = max_right
+            elif z.x0 >= env.x1:
+                min_left = max(z.x0, span_right + GAP_FROM_TEXT_BLOCKS)
+                if z.x1 - min_left < 40.0:
+                    continue
+                x_positions = [z.x1 - w]
+                min_x0 = min_left
             else:
-                # top/bottom bands: keep near left/right depending on target
-                if target_center.x < pr.width / 2:
-                    x0 = z.x0
+                x_positions = [z.x0, z.x1 - w]
+
+            for x0 in x_positions:
+                x0 = max(z.x0, min(x0, z.x1 - w))
+                if z.x1 <= env.x0:
+                    x1 = min(z.x1, x0 + w, max_x1)
+                elif z.x0 >= env.x1:
+                    x0 = max(x0, min_x0)
                     x1 = min(z.x1, x0 + w)
                 else:
-                    x1 = z.x1
-                    x0 = max(z.x0, x1 - w)
-
-            cand = fitz.Rect(x0, y0, x1, y1)
-
-            safe = (not _intersects_any(cand, blockers)) and (not _intersects_any(cand, occupied_buf))
-
-            # connector cost: fewer crossings then shortest
-            # obstacles = text blocks + other targets + occupied
-            obstacles: List[fitz.Rect] = []
-            for b in _page_text_blocks(page):
-                obstacles.append(inflate_rect(b, 1.5))
-            for oc in occupied:
-                obstacles.append(inflate_rect(oc, 2.0))
-            for t in targets:
-                obstacles.append(inflate_rect(t, 2.5))
-
-            # approximate best straight-line (using target union)
-            s, e = _straight_connector_best_pair(cand, target_union, obstacles)
-            crossings = 0
-            for ob in obstacles:
-                if _segment_hits_rect(s, e, ob):
-                    crossings += 1
-            length = math.hypot(e.x - s.x, e.y - s.y)
-
-            # score: must be safe first; then crossings; then length; then keep close to target_y
-            score = (0 if safe else 1e9) + crossings * 5000.0 + length + abs((_center(cand).y) - target_y) * 0.8
-
-            candidates.append((score, cand, wrapped, fs, safe))
+                    if span and not (
+                        x0 + w <= span_left - GAP_FROM_TEXT_BLOCKS
+                        or x0 >= span_right + GAP_FROM_TEXT_BLOCKS
+                    ):
+                        continue
+                    x1 = min(z.x1, x0 + w)
+                if x1 - x0 < 20.0:
+                    continue
+                cand = fitz.Rect(x0, y0, x1, y1)
+
+                safe = (not _intersects_any(cand, blockers)) and (not _intersects_any(cand, occupied_buf))
+
+                # connector cost: fewer crossings then shortest
+                # obstacles = text blocks + other targets + occupied
+                obstacles: List[fitz.Rect] = []
+                for b in _page_text_rects(page):
+                    obstacles.append(inflate_rect(b, 1.5))
+                for oc in occupied:
+                    obstacles.append(inflate_rect(oc, 2.0))
+                for t in targets:
+                    obstacles.append(inflate_rect(t, 2.5))
+
+                # approximate best straight-line (using target union)
+                s, e = _straight_connector_best_pair(cand, target_union, obstacles)
+                crossings = 0
+                for ob in obstacles:
+                    if _segment_hits_rect(s, e, ob):
+                        crossings += 1
+                length = math.hypot(e.x - s.x, e.y - s.y)
+
+                # score: must be safe first; then crossings; then length; then keep close to target_y/target_x
+                score = (
+                    (0 if safe else 1e9)
+                    + crossings * 5000.0
+                    + length
+                    + abs((_center(cand).y) - target_y) * 0.8
+                    + abs((_center(cand).x) - target_center.x) * 0.2
+                )
+
+                candidates.append((score, cand, wrapped, fs, safe))
 
     candidates.sort(key=lambda x: x[0])
     if not candidates:
         # emergency fallback
         fs, wrapped, w, h = _optimize_layout_for_margin(label, 140.0)
         fallback = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h)
         return fallback, wrapped, fs, False
 
+    for _, cand, wrapped, fs, safe in candidates:
+        if safe:
+            return cand, wrapped, fs, True
+
     _, cand, wrapped, fs, safe = candidates[0]
     return cand, wrapped, fs, safe
 
 # ============================================================
 # Search helpers (robust quote matching)
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
@@ -502,51 +581,51 @@ def annotate_pdf_bytes(
         # Box targets
         for t in targets:
             page1.draw_rect(t, color=RED, width=BOX_WIDTH)
         total_meta_hits += len(targets)
 
         # Place callout in margins/top/bottom ONLY
         callout_rect, wrapped_text, fs, safe = _choose_best_spot_margins_only(
             page1, targets, occupied_callouts, label
         )
 
         # White background only if safe (won’t cover text in allowed zones anyway, but keep the logic)
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
 
         # Obstacles for connector scoring (we still draw one straight line)
         obstacles: List[fitz.Rect] = []
-        for b in _page_text_blocks(page1):
+        for b in _page_text_rects(page1):
             obstacles.append(inflate_rect(b, 1.5))
         for oc in occupied_callouts:
             obstacles.append(inflate_rect(oc, 2.0))
         # treat all target boxes as obstacles (except the one we connect to is okay; we pull back anyway)
         expanded_all = [inflate_rect(t, 2.5) for t in targets]
 
         def connect_to(rect: fitz.Rect):
             obs = obstacles[:]
             for ot in expanded_all:
                 if not ot.intersects(rect):
                     obs.append(ot)
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
