"""
OpenAI Responses API Integration (New System)
Replaces deprecated Assistants API with Responses API
Uses web_search tool for evidence research
"""

import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from openai import OpenAI


def _get_secret(name: str):
    """Get secret from Streamlit or environment"""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


SEARCH_CONFIGS: Dict[str, Dict[str, int]] = {
    "1": {"pool": 50, "min": 1, "max": 5},
    "2_past": {"pool": 100, "min": 10, "max": 10},
    "2_future": {"pool": 100, "min": 10, "max": 10},
    "3": {"pool": 100, "min": 10, "max": 15},
    "4_past": {"pool": 100, "min": 10, "max": 10},
    "4_future": {"pool": 100, "min": 10, "max": 10},
    "5": {"pool": 50, "min": 3, "max": 10},
    "6": {"pool": 50, "min": 3, "max": 10},
    "7": {"pool": 20, "min": 3, "max": 5}
}

DEFAULT_SEARCH_CONFIG = {"pool": 50, "min": 8, "max": 10}

TIER_1_DOMAINS = {
    "grammy.com",
    "kennedy-center.org",
    "pulitzer.org",
    "carnegiehall.org",
    "metopera.org",
    "berliner-philharmoniker.de",
    "nytimes.com",
    "theguardian.com",
    "telegraph.co.uk",
    "gramophone.co.uk",
    "opera-news.com"
}

TIER_2_DOMAINS = {
    "latimes.com",
    "chicagotribune.com",
    "variety.com",
    "billboard.com",
    "classical-music.com",
    "salzburgfestival.at",
    "glyndebourne.com"
}

DIRECTORY_DOMAINS = {
    "operabase.com",
    "bachtrack.com"
}

SALARY_DOMAINS = {
    "bls.gov",
    "onetonline.org",
    "data.bls.gov"
}


def get_search_config(criterion_id: str) -> Dict[str, int]:
    config = SEARCH_CONFIGS.get(criterion_id, DEFAULT_SEARCH_CONFIG)
    return {
        "pool": config["pool"],
        "min": config["min"],
        "max": config["max"]
    }


def _normalize_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path.split("/")[0]
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _extract_year(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def _extract_venue_or_city(text: str) -> str:
    """Extract likely venue or city for diversity dedupe."""
    if not text:
        return ""
    t = (text or "").lower()
    # Common venue/city patterns
    cities = ["paris", "london", "vienna", "munich", "rome", "milan", "new york", "detroit", "chicago", "berlin", "moscow", "santa fe", "verona", "salzburg", "glyndebourne"]
    for c in cities:
        if c in t:
            return c
    # Opera houses / venues
    venues = ["metropolitan opera", "met opera", "carnegie hall", "bolshoi", "opéra", "opera", "staatsoper", "symphony", "orchestra", "arena"]
    for v in venues:
        if v in t:
            return v
    return ""


def _is_review(item: Dict) -> bool:
    text = f"{item.get('title', '')} {item.get('excerpt', '')}".lower()
    return "review" in text or "critically acclaimed" in text


def _is_announcement(item: Dict) -> bool:
    text = f"{item.get('title', '')} {item.get('excerpt', '')}".lower()
    return any(keyword in text for keyword in ["announcement", "season", "calendar", "press release", "program", "programme"])


def _is_sales_signal(item: Dict) -> bool:
    text = f"{item.get('title', '')} {item.get('excerpt', '')}".lower()
    return any(keyword in text for keyword in ["sold out", "box office", "chart", "stream", "ticket sales", "capacity"])


def _is_award_signal(item: Dict) -> bool:
    text = f"{item.get('title', '')} {item.get('excerpt', '')}".lower()
    return any(keyword in text for keyword in ["award", "prize", "honor", "honour", "fellowship", "medal"])


def _get_usage_from_response(response) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0, 0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
    return input_tokens, output_tokens, total_tokens


def _log_token_usage(criterion_id: str, response, retrieval_pool_size: int, max_results: int, min_results: int, strict_mode: bool) -> None:
    input_tokens, output_tokens, total_tokens = _get_usage_from_response(response)
    if total_tokens == 0:
        return
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "criterion_id": criterion_id,
        "retrieval_pool_size": retrieval_pool_size,
        "min_results": min_results,
        "max_results": max_results,
        "strict_mode": strict_mode,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }
    try:
        import streamlit as st
        if "token_usage_log" not in st.session_state:
            st.session_state.token_usage_log = []
        st.session_state.token_usage_log.append(log_entry)
    except Exception:
        pass


def _parse_search_results(content_text: str) -> List[Dict]:
    if not content_text:
        raise RuntimeError("API returned empty response")
    content_text = content_text.strip()

    start = content_text.find('[')
    end = content_text.rfind(']') + 1

    if start == -1 or end == 0:
        try:
            results = json.loads(content_text)
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")
        except json.JSONDecodeError:
            raise ValueError(
                f"No JSON array found in response. "
                f"Response was: {content_text[:200]}..."
            )
    else:
        json_str = content_text[start:end]
        results = json.loads(json_str)
        if not isinstance(results, list):
            raise ValueError("Response is not a JSON array")

    normalized_results = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if 'url' not in item or not item['url']:
            continue
        normalized_results.append({
            'url': item.get('url', ''),
            'title': item.get('title', 'Untitled'),
            'source': item.get('source', 'Unknown'),
            'excerpt': item.get('excerpt', ''),
            'relevance': item.get('relevance', '')
        })

    return normalized_results


def _get_existing_results(criterion_id: str) -> Tuple[set, set]:
    try:
        import streamlit as st
        existing = st.session_state.research_results.get(criterion_id, [])
        urls = {item.get("url", "") for item in existing if item.get("url")}
        domains = {_normalize_domain(url) for url in urls if url}
        return urls, domains
    except Exception:
        return set(), set()


def _rank_and_select_results(
    criterion_id: str,
    results: List[Dict],
    min_results: int,
    max_results: int
) -> List[Dict]:
    if not results:
        return []

    existing_c3_urls, existing_c3_domains = _get_existing_results("3")
    review_overlap_present = bool(existing_c3_urls or existing_c3_domains)
    any_non_directory = any(
        _normalize_domain(item.get("url", "")) not in DIRECTORY_DOMAINS
        for item in results
    )

    candidates = []
    seen_urls = set()
    for item in results:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        domain = _normalize_domain(url)
        source_norm = (item.get("source") or "").strip().lower() or domain
        year = _extract_year(f"{item.get('title', '')} {item.get('excerpt', '')} {url}")
        is_review = _is_review(item)
        is_announcement = _is_announcement(item)
        is_directory = domain in DIRECTORY_DOMAINS
        base_score = 1.0
        if domain in TIER_1_DOMAINS:
            base_score += 2.0
        elif domain in TIER_2_DOMAINS:
            base_score += 1.0
        if is_directory:
            base_score -= 0.6
            if criterion_id in {"2_past", "2_future"}:
                base_score -= 0.6
            if any_non_directory:
                base_score -= 0.8

        if criterion_id in {"2_past", "2_future", "3"} and is_review:
            base_score += 0.7
        if criterion_id in {"4_past", "4_future"} and is_announcement:
            base_score += 0.6
        if criterion_id == "5":
            if _is_sales_signal(item):
                base_score += 0.6
            if _is_award_signal(item):
                base_score += 0.3
        if criterion_id == "6" and _is_award_signal(item):
            base_score += 0.4
        if criterion_id == "7" and domain in SALARY_DOMAINS:
            base_score += 1.0

        overlap_with_c3 = url in existing_c3_urls or domain in existing_c3_domains
        if criterion_id in {"2_past", "2_future"} and overlap_with_c3:
            base_score -= 1.0
        if criterion_id == "6" and overlap_with_c3 and not _is_award_signal(item):
            base_score -= 0.6

        performance_key = ""
        if criterion_id in {"2_past", "2_future", "4_past", "4_future"}:
            title_text = (item.get("title", "") or "").lower()
            excerpt_text = (item.get("excerpt", "") or "").lower()
            combined = f"{title_text} {excerpt_text}"
            # Normalize: remove years, dates, numbers
            combined = re.sub(r"\b(19|20)\d{2}\b", "", combined)
            combined = re.sub(r"\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*", "", combined, flags=re.I)
            combined = re.sub(r"\d+", "", combined)
            combined = re.sub(r"\s+", " ", combined).strip()
            prod_part = combined[:60]
            venue = _extract_venue_or_city(f"{title_text} {excerpt_text}")
            performance_key = f"{prod_part}|{venue}" if venue else prod_part

        candidates.append({
            "item": item,
            "domain": domain,
            "source_norm": source_norm,
            "year": year,
            "is_review": is_review,
            "is_announcement": is_announcement,
            "is_directory": is_directory,
            "overlap_with_c3": overlap_with_c3,
            "domain_year_key": f"{domain}:{year}" if year else domain,
            "performance_key": performance_key,
            "base_score": base_score,
            "used": False
        })

    domain_cap = 3 if criterion_id in {"2_past", "2_future", "3", "4_past", "4_future"} else None
    selected = []
    domain_counts = {}
    source_counts = {}
    year_counts = {}
    review_count = 0
    directory_count = 0
    performance_counts = {}
    used_domain_year = set()

    def _pick_next(allow_domain_cap: bool) -> Optional[int]:
        best_idx = None
        best_score = -1e9
        for idx, cand in enumerate(candidates):
            if cand["used"]:
                continue
            if criterion_id == "1" and cand["domain_year_key"] in used_domain_year:
                continue
            if allow_domain_cap and domain_cap and domain_counts.get(cand["domain"], 0) >= domain_cap:
                continue
            if criterion_id in {"2_past", "2_future"} and cand["is_directory"] and any_non_directory:
                if any(not c["used"] and not c["is_directory"] for c in candidates):
                    continue
            if criterion_id in {"2_past", "2_future"} and cand["is_directory"] and directory_count >= 2:
                continue
            if cand["performance_key"] and performance_counts.get(cand["performance_key"], 0) >= 1:
                if any(not c["used"] and c["performance_key"] != cand["performance_key"] for c in candidates):
                    continue
            if criterion_id in {"2_past", "2_future"} and cand["is_review"] and review_overlap_present:
                if review_count >= 5:
                    if any(not c["used"] and not c["is_review"] for c in candidates):
                        continue
            score = cand["base_score"]
            if cand["domain"] and cand["domain"] not in domain_counts:
                score += 0.4
            if cand["source_norm"] and cand["source_norm"] not in source_counts:
                score += 0.3
            if cand["year"] and cand["year"] not in year_counts:
                score += 0.2
            if criterion_id == "3" and cand["source_norm"] and cand["source_norm"] not in source_counts:
                score += 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    target_count = min(max_results, len(candidates))
    while len(selected) < target_count:
        next_idx = _pick_next(allow_domain_cap=True)
        if next_idx is None:
            break
        cand = candidates[next_idx]
        cand["used"] = True
        # Capture ranking factors at selection time
        factors = []
        if cand["domain"] and cand["domain"] not in domain_counts:
            factors.append("domain diversity")
        if cand["source_norm"] and cand["source_norm"] not in source_counts:
            factors.append("publisher diversity")
        if cand["year"] and cand["year"] not in year_counts:
            factors.append("year diversity")
        if criterion_id in {"2_past", "2_future", "4_past", "4_future"} and cand["performance_key"] and performance_counts.get(cand["performance_key"], 0) == 0:
            factors.append("distinct performance")
        cand["_selection_factors"] = factors
        selected.append(cand)
        domain_counts[cand["domain"]] = domain_counts.get(cand["domain"], 0) + 1
        source_counts[cand["source_norm"]] = source_counts.get(cand["source_norm"], 0) + 1
        if cand["year"]:
            year_counts[cand["year"]] = year_counts.get(cand["year"], 0) + 1
        if cand["is_review"]:
            review_count += 1
        if cand["is_directory"]:
            directory_count += 1
        if cand["performance_key"]:
            performance_counts[cand["performance_key"]] = performance_counts.get(cand["performance_key"], 0) + 1
        if criterion_id == "1":
            used_domain_year.add(cand["domain_year_key"])

    if len(selected) < min_results:
        while len(selected) < min_results:
            next_idx = _pick_next(allow_domain_cap=False)
            if next_idx is None:
                break
            cand = candidates[next_idx]
            cand["used"] = True
            cand["_selection_factors"] = cand.get("_selection_factors", []) or ["fallback: relaxed constraints"]
            selected.append(cand)

    if criterion_id == "3":
        distinct_publishers = len(source_counts)
        if distinct_publishers < 3:
            remaining = [c for c in candidates if not c["used"] and c["source_norm"] not in source_counts]
            while distinct_publishers < 3 and remaining:
                replacement = remaining.pop(0)
                if not selected:
                    break
                worst = min(selected, key=lambda c: c["base_score"])
                selected.remove(worst)
                selected.append(replacement)
                source_counts[replacement["source_norm"]] = source_counts.get(replacement["source_norm"], 0) + 1
                distinct_publishers = len(source_counts)

    # Attach explainability metadata to each item
    out = []
    for cand in selected[:max_results]:
        item = dict(cand["item"])
        badges = []
        if cand["is_directory"]:
            badges.append("Directory")
        elif cand["is_announcement"]:
            badges.append("Announcement")
        elif cand["is_review"]:
            badges.append("Review")
        else:
            badges.append("Primary")
        if cand["domain"] in TIER_1_DOMAINS:
            badges.append("Tier 1")
        elif cand["domain"] in TIER_2_DOMAINS:
            badges.append("Tier 2")
        item["_meta"] = {
            "badges": badges,
            "ranking_factors": cand.get("_selection_factors", []),
            "stage": "strict" if not cand["is_directory"] else "fallback",
            "is_directory": cand["is_directory"],
            "is_announcement": cand["is_announcement"],
            "is_review": cand["is_review"],
        }
        out.append(item)

    return out


# System prompt with detailed USCIS guidance and criterion-specific instructions
RESEARCH_SYSTEM_PROMPT = """You are a visa paralegal assistant researching O-1 visa evidence for artists.

CRITICAL OUTPUT REQUIREMENT:
You MUST return ONLY a valid JSON array. No explanations, no markdown, no code blocks, no preamble, no postamble.
Just the raw JSON array starting with [ and ending with ].

===========================================
CRITERION-SPECIFIC SEARCH GUIDANCE
===========================================

Criterion 1 (Awards):
CRITICAL RULES FOR CRITERION 1:
- Find OFFICIAL award announcements from the awarding body itself
- PRIMARY SOURCE REQUIRED: Grammy.com, award websites, official award organization sites
- For music awards: Grammy.com, Opus Klassik (classical), Mercury Prize, etc.
- AVOID: Third-party publications discussing the award (Forbes, biography pages, news articles)
- AVOID: Artist biographies, Wikipedia, management pages
- MUST BE: The actual award organization's website or official winner announcement
- OK to include multiple awards from the same issuer if they are different award-year pages
- Avoid multiple results for the same award-year page (no duplicate Grammy 2026 pages)
Example GOOD sources: grammy.com/awards/winners, kennedy-center.org/honors
Example BAD sources: forbes.com/artist-biography, wikipedia.org, artist-website.com/awards

Criterion 2_past (Past Lead Roles):
- Find PAST performance evidence: reviews, programs, venue announcements
- Prioritize: Major venue websites, festival archives, reviews in major publications
- MUST show artist in lead/starring role
- Look for: Carnegie Hall programs, Met Opera archives, festival reviews
- Dates must be in the PAST
- Prefer reviews that show critical acclaim for the performance
- Also include official announcements from distinguished organizations (venue/festival sites)
- Use directories (Operabase/Bachtrack) only as a fallback

Criterion 2_future (Future Lead Roles):
- Find UPCOMING performances: season brochures, venue calendars
- Prioritize: Official venue sites, Operabase, Bachtrack, major festival announcements
- MUST clearly show lead/starring role
- Look for: 2026-2027 season announcements, confirmed bookings
- Dates must be in the FUTURE
- Prefer official venue/festival announcements over directories
- Use directories (Operabase/Bachtrack) only as a fallback
- Avoid multiple listings for the same production or city when possible

Criterion 3 (Critical Reviews):
- Find SUBSTANTIAL reviews and articles about the artist
- Prioritize: New York Times, Guardian, Telegraph, major newspapers, respected arts magazines
- AVOID: Event listings, one-line mentions, low-quality blogs
- MUST BE: Full review articles (300+ words) focused on the artist's performance/work
- Look for: "Review:", critic bylines, detailed artistic analysis
- Aim for 10-15 results across 3 or more publishers when possible
- Avoid returning many results from a single publication

Criterion 4_past (Past Distinguished Organizations):
- Find evidence of PAST work with prestigious venues/ensembles
- Include: Performance evidence + organization prestige evidence
- Prioritize: Carnegie Hall, Metropolitan Opera, major symphony orchestras
- Look for: Official venue archives, program listings, performance announcements
- Dates must be in the PAST
- Prioritize official venue announcements/archives
- Use directories (Operabase/Bachtrack) only as a fallback

Criterion 4_future (Future Distinguished Organizations):
- Find ANNOUNCED future engagements with prestigious organizations
- Include: Announcement + organization prestige evidence
- Look for: Season brochures, official venue calendars, press releases
- Dates must be in the FUTURE
- Prioritize official venue announcements/season brochures
- Use directories (Operabase/Bachtrack) only as a fallback

Criterion 5 (Commercial Success):
- Find evidence of box office success, chart rankings, sold-out shows
- Prioritize: Billboard charts, Spotify charts, venue capacity data, ticket sales
- Look for: "sold out", chart positions, streaming numbers, box office reports
- Prefer the strongest evidence even if fewer results

Criterion 6 (Recognition):
- Find evidence of recognition from leading organizations/experts
- Include: Awards from organizations, expert testimonials, honors, fellowships
- Look for: Honorary degrees, fellowships, institutional recognition
- Aim for one piece of evidence per expert/critic/organization
- Avoid overlap with Criterion 3 unless discussing a distinct achievement (award/honor)

Criterion 7 (High Salary):
- Find evidence of artist fees/contracts (rare to find publicly)
- Include: BLS wage data, O*NET salary benchmarks, union scales
- Look for: Contract announcements, fee schedules, salary surveys
- Focus on reliable statistics sources; diversity is not required

===========================================
SOURCE QUALITY HIERARCHY
===========================================

TIER 1 - HIGHEST QUALITY (Always prioritize):
For Awards (Criterion 1):
- grammy.com, kennedy-center.org, pulitzer.org
- Official award organization websites ONLY
- NOT news coverage of awards - only the award body itself

For Performances (Criteria 2, 4):
- carnegiehall.org, metopera.org, berliner-philharmoniker.de
- Official venue websites, season brochures
- Operabase.com, Bachtrack.com (verified performance databases)

For Reviews (Criterion 3):
- nytimes.com, theguardian.com, telegraph.co.uk
- Major newspaper arts sections
- gramophone.co.uk, opera-news.com (specialist publications)

TIER 2 - GOOD (Use if Tier 1 limited):
- Regional major newspapers (LA Times, Chicago Tribune)
- Industry publications (Variety, Billboard, Classical Music Magazine)
- Major festival websites (Salzburg, Glyndebourne)

NEVER USE (Even if they mention the artist):
- Artist's personal website or biography page
- Management company pages
- Wikipedia (use sources it cites instead, but NOT the Wikipedia page itself)
- Forbes lists or celebrity net worth sites
- Social media posts
- Fan sites or personal blogs
- Generic "famous musicians" lists

===========================================
USCIS REGULATORY STANDARDS
===========================================

According to 8 CFR 214.2(o)(3)(iv):

Key Standards:
✅ Evidence must be from CREDIBLE, AUTHORITATIVE sources
✅ PRIMARY sources preferred (official organizations, not news coverage)
✅ SUBSTANTIAL evidence (full articles, not mentions)
✅ DOCUMENTED (verifiable, with proper attribution)

Grammy Awards Exception:
- Grammy.com automatically satisfies awards criterion
- No additional prestige evidence needed
- But MUST be from grammy.com itself

Other Awards:
- MUST be from official award organization website
- News articles ABOUT the award don't count as primary evidence
- Need to prove award's prestige separately

===========================================
YOUR TASK
===========================================

Search the web broadly to build a large pool of sources that support the given O-1 criterion.
We will rank and diversify the pool after retrieval, so include Tier 2/3 sources when Tier 1 is limited.

CRITICAL FILTERING:
- For Criterion 1: ONLY include grammy.com or official award organization sites
- For Criterion 1: EXCLUDE Forbes, biographies, news articles about awards
- For all criteria: Verify source matches the tier hierarchy above
- For all criteria: Ensure source type matches criterion requirements

OUTPUT FORMAT - ABSOLUTELY CRITICAL:
Return ONLY a JSON array. No other text whatsoever.

Correct format:
[{"url":"https://grammy.com/winners/yo-yo-ma","title":"Yo-Yo Ma - Grammy Winners","source":"Grammy.com","excerpt":"18-time Grammy Award winner","relevance":"Official Grammy website showing multiple awards"}]

WRONG - Do not include explanatory text:
Here are the results:
[...]

WRONG - Do not include markdown:
```json
[...]
```

ONLY return the raw JSON array with no additional text before or after.
"""


def search_with_responses_api(
    artist_name: str,
    criterion_id: str,
    criterion_description: str,
    name_variants: Optional[List[str]] = None,
    artist_field: Optional[str] = None,
    feedback: Optional[str] = None,
    max_results: Optional[int] = None,
    min_results: Optional[int] = None,
    retrieval_pool_size: Optional[int] = None,
    relaxation_stage: Optional[str] = None
) -> List[Dict]:
    """
    Use OpenAI Responses API with web_search tool for evidence research
    
    Args:
        artist_name: Beneficiary name
        criterion_id: Which criterion (e.g., "1", "3", "2_past")
        criterion_description: Full description of the criterion
        name_variants: Alternative spellings of artist name
        artist_field: Field of work (e.g., "Classical Music")
        feedback: User feedback for regeneration
        max_results: Maximum number of results to return
        min_results: Minimum number of results to return
        retrieval_pool_size: Size of the broad retrieval pool
    
    Returns:
        List of evidence sources with url, title, source, excerpt, relevance
    """
    
    # Get API key
    api_key = _get_secret("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets")
    
    client = OpenAI(api_key=api_key)
    
    config = get_search_config(criterion_id)
    if max_results is None:
        max_results = config["max"]
    if min_results is None:
        min_results = config["min"]
    if retrieval_pool_size is None:
        retrieval_pool_size = config["pool"]

    max_results = int(max_results)
    min_results = int(min_results)
    retrieval_pool_size = int(max(retrieval_pool_size, max_results))

    # Build the research prompt
    prompt = f"""Search the web for evidence that {artist_name} meets this O-1 visa criterion:

Criterion ({criterion_id}): {criterion_description}

Artist: {artist_name}
"""
    
    if name_variants:
        prompt += f"Also known as: {', '.join(name_variants)}\n"
    
    if artist_field:
        prompt += f"Field: {artist_field}\n"
    
    # Add criterion-specific filtering instructions
    if criterion_id == "1":
        prompt += f"""
CRITICAL FOR CRITERION 1 (AWARDS):
- ONLY include sources from official award organization websites
- For music: grammy.com, opusklassik.de, mercuryprize.com, etc.
- DO NOT include Forbes, Wikipedia, artist biographies, or news articles
- DO NOT include third-party coverage of awards
- MUST be the award organization's own website
- OK to include multiple awards from the same issuer if they are different award-year pages
- Avoid multiple results for the same award-year page

REJECT these source types:
- forbes.com/artist-name
- wikipedia.org
- artist-website.com/awards
- news articles ABOUT awards (nytimes.com/yo-yo-ma-wins-grammy)
- biography pages
- management/publicity sites

ACCEPT only these source types:
- grammy.com/awards/winners
- kennedy-center.org/honors
- pulitzer.org/winners
- opusklassik.de/preistraeger (for classical music)
- [official-award-organization].org/winners
"""
    elif criterion_id == "6":
        prompt += f"""
CRITICAL FOR CRITERION 6 (RECOGNITION):
- Focus on recognition from leading organizations, institutions, or experts
- Include: Honorary degrees, fellowships, institutional awards, expert testimonials
- Prioritize: Universities, professional associations, government entities
"""
    elif criterion_id == "7":
        prompt += f"""
CRITICAL FOR CRITERION 7 (HIGH SALARY):
- Artist fee/contract data is rarely public - don't force it
- Include: BLS wage data, O*NET salary benchmarks, union scales
- If no reliable fee data exists, return fewer sources
"""
    
    prompt += f"\nFind UP TO {retrieval_pool_size} sources for a broad retrieval pool.\n"
    prompt += "Prioritize quality, but include Tier 2/3 sources if Tier 1 is limited.\n"
    prompt += "Avoid duplicate results from the same source whenever possible.\n"
    
    if feedback:
        prompt += f"\nUser feedback: {feedback}\n"

    prompt += "\nIf this is a relaxed fallback search, allow directories and broader sources only when official sources are insufficient.\n"
    
    prompt += """
Return ONLY a JSON array in this format:
[
  {
    "url": "https://example.com/article",
    "title": "Article Title",
    "source": "Publication Name",
    "excerpt": "Brief relevant excerpt",
    "relevance": "Why this supports the criterion"
  }
]

DO NOT include any other text. ONLY the JSON array.
"""

    def _run_search(strict_mode: bool) -> List[Dict]:
        try:
            search_prompt = prompt
            if strict_mode:
                search_prompt += "\nSTRICT MODE: favor Tier 1/2 and avoid directories unless unavoidable.\n"
            else:
                search_prompt += "\nRELAXED MODE: allow directories and Tier 3 sources if needed to meet minimum results.\n"

            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": search_prompt}
                ],
                tools=[
                    {
                        "type": "web_search_preview_2025_03_11"
                    }
                ]
            )
            _log_token_usage(criterion_id, response, retrieval_pool_size, max_results, min_results, strict_mode)
            content_text = response.output_text
            return _parse_search_results(content_text)
        except Exception as e:
            raise RuntimeError(f"OpenAI Responses API error: {str(e)}")

    try:
        # Determine search strategy from relaxation_stage
        if relaxation_stage == "strict":
            results = _run_search(strict_mode=True)
        elif relaxation_stage == "relaxed":
            results = _run_search(strict_mode=False)
        else:
            results = _run_search(strict_mode=True)
            if len(results) < min_results:
                fallback_results = _run_search(strict_mode=False)
                results = results + [item for item in fallback_results if item.get("url") not in {r.get("url") for r in results}]

        ranked_results = _rank_and_select_results(criterion_id, results, min_results, max_results)
        if not ranked_results and results:
            ranked_results = results[:max_results]
        return ranked_results
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to parse API response as JSON: {str(e)}"
        )


# ============================================================
# Helper function for batch searching multiple criteria
# ============================================================

def batch_search_with_responses(
    artist_name: str,
    criteria_ids: List[str],
    criteria_descriptions: Dict[str, str],
    name_variants: Optional[List[str]] = None,
    artist_field: Optional[str] = None,
    max_results_per_criterion: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Search multiple criteria in sequence using Responses API
    
    Returns:
        {criterion_id: [results], ...}
    """
    
    all_results = {}
    
    for cid in criteria_ids:
        desc = criteria_descriptions.get(cid, "")
        
        try:
            results = search_with_responses_api(
                artist_name=artist_name,
                criterion_id=cid,
                criterion_description=desc,
                name_variants=name_variants,
                artist_field=artist_field,
                max_results=max_results_per_criterion
            )
            all_results[cid] = results
        
        except Exception as e:
            # Log error but continue with other criteria
            print(f"Error searching criterion {cid}: {str(e)}")
            all_results[cid] = []
    
    return all_results
