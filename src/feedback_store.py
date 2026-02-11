import json
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple, Optional


SCHEMA_VERSION = 1


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _feedback_dir() -> str:
    return os.path.join(_project_root(), "data", "feedback")


def _slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "case"


def _default_feedback() -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "last_updated": None,
        "last_feedback_hash": "",
        "global_rules": {"search": [], "highlight": []},
        "per_criterion_rules": {},
        "rejected_sources_by_criterion": {},
        "approved_quotes_by_criterion": {},
        "rejected_quotes_by_criterion": {},
        "ranking_rationale": {"rule_based": [], "model": ""},
        "initial_logic_current": "",
        "initial_logic_history": []
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_feedback_paths(beneficiary_name: str) -> Tuple[str, str]:
    base_dir = _feedback_dir()
    _ensure_dir(base_dir)
    global_path = os.path.join(base_dir, "global.json")
    case_path = os.path.join(base_dir, f"{_slugify(beneficiary_name)}.json")
    return global_path, case_path


def load_feedback(path: str) -> Dict:
    if not os.path.exists(path):
        return _default_feedback()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, dict):
                return _default_feedback()
            return data
    except Exception:
        return _default_feedback()


def save_feedback(path: str, data: Dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _merge_rule_lists(a: List[Dict], b: List[Dict]) -> List[Dict]:
    merged = []
    seen = set()
    for rule in a + b:
        key = (
            rule.get("target"),
            rule.get("action"),
            rule.get("subject"),
            rule.get("criterion_id"),
            rule.get("why")
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(rule)
    return merged


def _merge_dict_of_lists(a: Dict, b: Dict) -> Dict:
    out = deepcopy(a)
    for key, value in b.items():
        if key not in out:
            out[key] = deepcopy(value)
        else:
            existing = out.get(key, [])
            combined = existing + value
            deduped = []
            seen = set()
            for item in combined:
                marker = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(item)
            out[key] = deduped
    return out


def merge_feedback(global_feedback: Dict, case_feedback: Dict) -> Dict:
    merged = _default_feedback()
    merged["global_rules"]["search"] = _merge_rule_lists(
        global_feedback.get("global_rules", {}).get("search", []),
        case_feedback.get("global_rules", {}).get("search", [])
    )
    merged["global_rules"]["highlight"] = _merge_rule_lists(
        global_feedback.get("global_rules", {}).get("highlight", []),
        case_feedback.get("global_rules", {}).get("highlight", [])
    )

    merged["per_criterion_rules"] = _merge_dict_of_lists(
        global_feedback.get("per_criterion_rules", {}),
        case_feedback.get("per_criterion_rules", {})
    )

    merged["rejected_sources_by_criterion"] = _merge_dict_of_lists(
        global_feedback.get("rejected_sources_by_criterion", {}),
        case_feedback.get("rejected_sources_by_criterion", {})
    )
    merged["approved_quotes_by_criterion"] = _merge_dict_of_lists(
        global_feedback.get("approved_quotes_by_criterion", {}),
        case_feedback.get("approved_quotes_by_criterion", {})
    )
    merged["rejected_quotes_by_criterion"] = _merge_dict_of_lists(
        global_feedback.get("rejected_quotes_by_criterion", {}),
        case_feedback.get("rejected_quotes_by_criterion", {})
    )

    merged["ranking_rationale"] = case_feedback.get("ranking_rationale") or global_feedback.get("ranking_rationale") or {}
    merged["initial_logic_current"] = case_feedback.get("initial_logic_current") or global_feedback.get("initial_logic_current") or ""
    merged["initial_logic_history"] = case_feedback.get("initial_logic_history") or global_feedback.get("initial_logic_history") or []
    return merged


def _distill_comment_to_rules(comment: str, criterion_id: str, target: str) -> List[Dict]:
    if not comment:
        return []
    text = comment.lower()
    rules = []

    def add_rule(action: str, subject: str, why: str) -> None:
        rules.append({
            "target": target,
            "action": action,
            "subject": subject,
            "criterion_id": criterion_id or "all",
            "why": why,
            "source": "regenerate_comment",
            "raw_comment": comment.strip(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    if "avoid" in text:
        add_rule("avoid", "low_quality_sources", "User requested to avoid certain sources")
    if "review" in text:
        add_rule("prefer", "reviews", "User emphasized reviews")
    if "official" in text or "announcement" in text or "venue" in text:
        add_rule("prefer", "official_announcements", "User emphasized official sources")
    if "primary" in text:
        add_rule("prefer", "primary_sources", "User emphasized primary sources over secondary")
    if "diversity" in text or "publisher" in text:
        add_rule("emphasize", "publisher_diversity", "User requested diverse publishers")
    if "geograph" in text or "international" in text or "location" in text:
        add_rule("emphasize", "geographic_diversity", "User requested geographic diversity")
    if "production" in text or "same production" in text or "different performance" in text:
        add_rule("emphasize", "production_diversity", "User requested distinct productions, not same show on multiple dates")
    if "venue" in text and ("diversity" in text or "different" in text):
        add_rule("emphasize", "venue_diversity", "User requested diverse venues")
    if "directory" in text or "operabase" in text or "bachtrack" in text:
        add_rule("avoid", "directories", "User requested to avoid directory sites")
    if "award" in text:
        add_rule("prefer", "awards", "User emphasized awards")
    if "sold out" in text or "box office" in text or "stream" in text:
        add_rule("prefer", "commercial_success", "User emphasized commercial success evidence")

    if not rules:
        add_rule("emphasize", "quality", "General quality emphasis from user comment")

    return rules


def _build_rule_based_rationale(feedback: Dict) -> List[str]:
    rationale = []
    global_rules = feedback.get("global_rules", {})
    per_criterion_rules = feedback.get("per_criterion_rules", {})
    if global_rules.get("search"):
        rationale.append("Global search rules apply to all criteria.")
    if global_rules.get("highlight"):
        rationale.append("Global highlight rules apply to all criteria.")
    if per_criterion_rules:
        rationale.append("Criterion-specific rules override global rules when present.")
    return rationale


def _build_model_rationale_summary(feedback: Dict) -> str:
    rules = feedback.get("global_rules", {}).get("search", []) + feedback.get("global_rules", {}).get("highlight", [])
    if not rules:
        return "No specific feedback rules have been recorded yet."
    subjects = sorted({r.get("subject") for r in rules if r.get("subject")})
    return (
        "Recent feedback suggests emphasizing "
        + ", ".join(subjects[:6])
        + " while avoiding low-quality or non-authoritative sources."
    )


def _build_initial_logic_snapshot() -> str:
    return (
        "Search baseline: broad retrieval pool per criterion, strict pass then relaxed fallback, "
        "rank by prestige + diversity, cap per-domain clustering. "
        "Highlight baseline: extract short, high-signal quotes that directly support selected criteria."
    )


def _maybe_update_snapshot(feedback: Dict) -> None:
    snapshot = _build_initial_logic_snapshot()
    if feedback.get("initial_logic_current") != snapshot:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logic": snapshot
        }
        history = feedback.get("initial_logic_history", [])
        history.append(entry)
        feedback["initial_logic_history"] = history
        feedback["initial_logic_current"] = snapshot


def _update_rationale(feedback: Dict) -> None:
    feedback["ranking_rationale"] = {
        "rule_based": _build_rule_based_rationale(feedback),
        "model": _build_model_rationale_summary(feedback)
    }


def _touch(feedback: Dict) -> None:
    feedback["last_updated"] = datetime.utcnow().isoformat() + "Z"


def update_feedback_for_research(
    beneficiary_name: str,
    criterion_id: str,
    rejected_urls: List[str],
    regenerate_comment: Optional[str]
) -> None:
    global_path, case_path = get_feedback_paths(beneficiary_name)
    global_feedback = load_feedback(global_path)
    case_feedback = load_feedback(case_path)

    if regenerate_comment:
        rules = _distill_comment_to_rules(regenerate_comment, criterion_id, "search")
        global_feedback.setdefault("per_criterion_rules", {}).setdefault(criterion_id, [])
        case_feedback.setdefault("per_criterion_rules", {}).setdefault(criterion_id, [])
        global_feedback["per_criterion_rules"][criterion_id] += rules
        case_feedback["per_criterion_rules"][criterion_id] += rules

    if rejected_urls:
        case_feedback.setdefault("rejected_sources_by_criterion", {})
        case_feedback["rejected_sources_by_criterion"].setdefault(criterion_id, [])
        for url in rejected_urls:
            case_feedback["rejected_sources_by_criterion"][criterion_id].append({"url": url})

        global_feedback.setdefault("rejected_sources_by_criterion", {})
        global_feedback["rejected_sources_by_criterion"].setdefault(criterion_id, [])
        for url in rejected_urls:
            domain = _extract_domain(url)
            if domain:
                global_feedback["rejected_sources_by_criterion"][criterion_id].append({"domain": domain})

    _maybe_update_snapshot(global_feedback)
    _maybe_update_snapshot(case_feedback)
    _update_rationale(global_feedback)
    _update_rationale(case_feedback)
    _touch(global_feedback)
    _touch(case_feedback)

    save_feedback(global_path, global_feedback)
    save_feedback(case_path, case_feedback)


def update_feedback_for_highlight(
    beneficiary_name: str,
    criterion_id: str,
    approved_quotes: List[str],
    rejected_quotes: List[str],
    regenerate_comment: Optional[str]
) -> None:
    global_path, case_path = get_feedback_paths(beneficiary_name)
    global_feedback = load_feedback(global_path)
    case_feedback = load_feedback(case_path)

    if regenerate_comment:
        rules = _distill_comment_to_rules(regenerate_comment, criterion_id, "highlight")
        global_feedback.setdefault("per_criterion_rules", {}).setdefault(criterion_id, [])
        case_feedback.setdefault("per_criterion_rules", {}).setdefault(criterion_id, [])
        global_feedback["per_criterion_rules"][criterion_id] += rules
        case_feedback["per_criterion_rules"][criterion_id] += rules

    if approved_quotes:
        case_feedback.setdefault("approved_quotes_by_criterion", {})
        case_feedback["approved_quotes_by_criterion"].setdefault(criterion_id, [])
        case_feedback["approved_quotes_by_criterion"][criterion_id] += approved_quotes

    if rejected_quotes:
        case_feedback.setdefault("rejected_quotes_by_criterion", {})
        case_feedback["rejected_quotes_by_criterion"].setdefault(criterion_id, [])
        case_feedback["rejected_quotes_by_criterion"][criterion_id] += rejected_quotes

    _maybe_update_snapshot(global_feedback)
    _maybe_update_snapshot(case_feedback)
    _update_rationale(global_feedback)
    _update_rationale(case_feedback)
    _touch(global_feedback)
    _touch(case_feedback)

    save_feedback(global_path, global_feedback)
    save_feedback(case_path, case_feedback)


def get_merged_feedback(beneficiary_name: str) -> Dict:
    global_path, case_path = get_feedback_paths(beneficiary_name)
    global_feedback = load_feedback(global_path)
    case_feedback = load_feedback(case_path)
    return merge_feedback(global_feedback, case_feedback)


def build_search_feedback_text(merged_feedback: Dict, criterion_id: str) -> str:
    lines = []
    global_rules = merged_feedback.get("global_rules", {}).get("search", [])
    if global_rules:
        lines.append("Global search preferences:")
        for rule in global_rules[:8]:
            lines.append(f"- {rule.get('action')} {rule.get('subject')}: {rule.get('why')}")

    per_rules = merged_feedback.get("per_criterion_rules", {}).get(criterion_id, [])
    if per_rules:
        lines.append(f"Criterion ({criterion_id}) preferences:")
        for rule in per_rules[:8]:
            lines.append(f"- {rule.get('action')} {rule.get('subject')}: {rule.get('why')}")

    rejected = merged_feedback.get("rejected_sources_by_criterion", {}).get(criterion_id, [])
    if rejected:
        domains = [r.get("domain") for r in rejected if r.get("domain")]
        urls = [r.get("url") for r in rejected if r.get("url")]
        if domains:
            lines.append("Avoid domains: " + ", ".join(sorted(set(domains))[:8]))
        if urls:
            lines.append("Avoid URLs like: " + ", ".join(urls[:5]))

    return "\n".join(lines).strip()


def build_highlight_feedback_payload(merged_feedback: Dict, criterion_id: str) -> Dict:
    approved = merged_feedback.get("approved_quotes_by_criterion", {}).get(criterion_id, [])
    rejected = merged_feedback.get("rejected_quotes_by_criterion", {}).get(criterion_id, [])
    return {
        "approved_examples": approved[:10],
        "rejected_examples": rejected[:10]
    }


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        parts = url.split("/")
        domain = parts[2] if len(parts) > 2 else parts[0]
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""
