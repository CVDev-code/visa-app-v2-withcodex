"""
OpenAI Responses API Integration (New System)
Replaces deprecated Assistants API with Responses API
Uses web_search tool for evidence research
"""

import json
import os
from typing import List, Dict, Optional
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
Example GOOD sources: grammy.com/awards/winners, kennedy-center.org/honors
Example BAD sources: forbes.com/artist-biography, wikipedia.org, artist-website.com/awards

Criterion 2_past (Past Lead Roles):
- Find PAST performance evidence: reviews, programs, venue announcements
- Prioritize: Major venue websites, festival archives, reviews in major publications
- MUST show artist in lead/starring role
- Look for: Carnegie Hall programs, Met Opera archives, festival reviews
- Dates must be in the PAST

Criterion 2_future (Future Lead Roles):
- Find UPCOMING performances: season brochures, venue calendars
- Prioritize: Official venue sites, Operabase, Bachtrack, major festival announcements
- MUST clearly show lead/starring role
- Look for: 2026-2027 season announcements, confirmed bookings
- Dates must be in the FUTURE

Criterion 3 (Critical Reviews):
- Find SUBSTANTIAL reviews and articles about the artist
- Prioritize: New York Times, Guardian, Telegraph, major newspapers, respected arts magazines
- AVOID: Event listings, one-line mentions, low-quality blogs
- MUST BE: Full review articles (300+ words) focused on the artist's performance/work
- Look for: "Review:", critic bylines, detailed artistic analysis

Criterion 4_past (Past Distinguished Organizations):
- Find evidence of PAST work with prestigious venues/ensembles
- Include: Performance evidence + organization prestige evidence
- Prioritize: Carnegie Hall, Metropolitan Opera, major symphony orchestras
- Look for: Official venue archives, program listings, performance announcements
- Dates must be in the PAST

Criterion 4_future (Future Distinguished Organizations):
- Find ANNOUNCED future engagements with prestigious organizations
- Include: Announcement + organization prestige evidence
- Look for: Season brochures, official venue calendars, press releases
- Dates must be in the FUTURE

Criterion 5 (Commercial Success):
- Find evidence of box office success, chart rankings, sold-out shows
- Prioritize: Billboard charts, Spotify charts, venue capacity data, ticket sales
- Look for: "sold out", chart positions, streaming numbers, box office reports

Criterion 6 (Recognition):
- Find evidence of recognition from leading organizations/experts
- Include: Awards from organizations, expert testimonials, honors, fellowships
- Look for: Honorary degrees, fellowships, institutional recognition

Criterion 7 (High Salary):
- Find evidence of artist fees/contracts (rare to find publicly)
- Include: BLS wage data, O*NET salary benchmarks, union scales
- Look for: Contract announcements, fee schedules, salary surveys

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

Search the web for 8-10 high-quality sources that support the given O-1 criterion.

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
    max_results: int = 10
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
    
    Returns:
        List of evidence sources with url, title, source, excerpt, relevance
    """
    
    # Get API key
    api_key = _get_secret("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets")
    
    client = OpenAI(api_key=api_key)
    
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

QUALITY OVER QUANTITY:
If you can only find 3-5 official award sites, return only those 3-5.
DO NOT fill the remaining slots with Forbes or biographies.
Better to return 3 perfect sources than 10 mixed sources.
"""
    elif criterion_id == "6":
        prompt += f"""
CRITICAL FOR CRITERION 6 (RECOGNITION):
- Focus on recognition from leading organizations, institutions, or experts
- Include: Honorary degrees, fellowships, institutional awards, expert testimonials
- Prioritize: Universities, professional associations, government entities

QUALITY OVER QUANTITY:
If you can only find 3-5 high-quality recognition sources, return only those.
DO NOT include generic articles or weak sources to fill the quota.
"""
    elif criterion_id == "7":
        prompt += f"""
CRITICAL FOR CRITERION 7 (HIGH SALARY):
- Artist fee/contract data is rarely public - don't force it
- Include: BLS wage data, O*NET salary benchmarks, union scales
- If no reliable fee data exists, return fewer sources

QUALITY OVER QUANTITY:
Salary data is often unavailable. Return 2-3 sources with benchmark data rather than 10 speculative sources.
"""
    
    # Adjust the "find X sources" instruction based on criterion
    if criterion_id in ["1", "6", "7"]:
        prompt += f"\nFind UP TO {max_results} high-quality sources (following the criterion-specific guidance above).\n"
        prompt += "Return FEWER sources if necessary to maintain quality standards.\n"
    else:
        prompt += f"\nFind {max_results} high-quality sources (following the criterion-specific guidance above).\n"
    
    if feedback:
        prompt += f"\nUser feedback: {feedback}\n"
    
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
    
    try:
        # Call Responses API with web_search tool
        response = client.responses.create(
            model="gpt-4o",  # Use gpt-4o for web search support
            input=[
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            tools=[
                {
                    "type": "web_search_preview_2025_03_11"
                    # Note: 'name' parameter not needed in Responses API
                }
            ]
        )
        
        # Extract text from response
        content_text = response.output_text
        
        if not content_text:
            raise RuntimeError("API returned empty response")
        
        # Parse JSON response
        try:
            # Try to extract JSON array from response
            # The response might include extra text since we can't force JSON format
            content_text = content_text.strip()
            
            # Try to find JSON array markers
            # Look for [ and ] that likely contain our JSON
            start = content_text.find('[')
            end = content_text.rfind(']') + 1
            
            if start == -1 or end == 0:
                # No JSON array found - try to parse the whole thing
                # Maybe it's just the JSON without extra text
                try:
                    results = json.loads(content_text)
                    if isinstance(results, list):
                        # Great, it was just a JSON array
                        pass
                    else:
                        raise ValueError("Response is not a JSON array")
                except json.JSONDecodeError:
                    raise ValueError(
                        f"No JSON array found in response. "
                        f"Response was: {content_text[:200]}..."
                    )
            else:
                # Found array markers - extract JSON
                json_str = content_text[start:end]
                results = json.loads(json_str)
                
                if not isinstance(results, list):
                    raise ValueError("Response is not a JSON array")
            
            # Validate and normalize results
            normalized_results = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                
                # Ensure required fields
                if 'url' not in item or not item['url']:
                    continue
                
                normalized_results.append({
                    'url': item.get('url', ''),
                    'title': item.get('title', 'Untitled'),
                    'source': item.get('source', 'Unknown'),
                    'excerpt': item.get('excerpt', ''),
                    'relevance': item.get('relevance', '')
                })
            
            return normalized_results[:max_results]
        
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to parse API response as JSON: {str(e)}\n\n"
                f"Response was:\n{content_text[:500]}"
            )
    
    except Exception as e:
        raise RuntimeError(f"OpenAI Responses API error: {str(e)}")


# ============================================================
# Helper function for batch searching multiple criteria
# ============================================================

def batch_search_with_responses(
    artist_name: str,
    criteria_ids: List[str],
    criteria_descriptions: Dict[str, str],
    name_variants: Optional[List[str]] = None,
    artist_field: Optional[str] = None,
    max_results_per_criterion: int = 10
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
