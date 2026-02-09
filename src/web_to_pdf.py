"""
Web scraping and PDF conversion for O-1 Research Assistant.

This module fetches webpages and converts them to clean PDFs with consistent margins
suitable for PDF annotation.

IMPROVEMENTS v14:
- Left-aligned publication logos
- Justified text (newspaper style)
- Globe icon for URL (🌐 or inline SVG)
- Double divider lines under URL (closer spacing)
- Font detection from original HTML
- Footer logo extraction
- Reduced header margins (0 0 10pt 0)
- Max image height limit (400pt)
"""

import io
import os
from typing import Dict, Optional, Tuple, List
from datetime import datetime

MIN_CONTENT_CHARS = 200
PLAYWRIGHT_TIMEOUT_MS = 20000


# ============================================================
# Translation Support
# ============================================================

def _detect_and_translate_content(content: str, html_content: str = "") -> Tuple[str, bool]:
    """
    Detect language and translate to English if needed.
    
    Supports multiple translation libraries with fallback:
    1. deep-translator (more reliable)
    2. googletrans (fallback)
    3. None (graceful degradation)
    
    Args:
        content: Text content to check/translate
        html_content: Optional HTML for better language detection
        
    Returns:
        (translated_content, was_translated)
    """
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        print("[Translation] langdetect not installed - translation disabled")
        print("[Translation] Install with: pip install langdetect")
        return content, False
    
    # Detect language
    detection_text = (html_content[:2000] if html_content else content[:2000]).strip()
    
    if not detection_text:
        return content, False
    
    try:
        detected_lang = detect(detection_text)
    except LangDetectException:
        # If detection fails, assume English
        return content, False
    
    print(f"[Translation] Detected language: {detected_lang}")
    
    # If already English, no translation needed
    if detected_lang == 'en':
        return content, False
    
    # Try deep-translator first (more reliable)
    try:
        from deep_translator import GoogleTranslator
        
        print(f"[Translation] Using deep-translator to translate from {detected_lang} to English...")
        
        # Split into chunks (API has length limits)
        max_chunk_size = 4000
        chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
        translated_chunks = []
        translator = GoogleTranslator(source=detected_lang, target='en')
        
        for i, chunk in enumerate(chunks):
            print(f"[Translation] Translating chunk {i+1}/{len(chunks)}...")
            try:
                result = translator.translate(chunk)
                translated_chunks.append(result)
            except Exception as e:
                print(f"[Translation] Error translating chunk {i+1}: {e}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        translated_content = '\n'.join(translated_chunks)
        print(f"[Translation] Successfully translated {len(chunks)} chunks with deep-translator")
        
        return translated_content, True
        
    except ImportError:
        print("[Translation] deep-translator not installed, trying googletrans...")
        
        # Fallback to googletrans
        try:
            from googletrans import Translator
            
            print(f"[Translation] Using googletrans to translate from {detected_lang} to English...")
            translator = Translator()
            
            # Split into chunks
            max_chunk_size = 4000
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"[Translation] Translating chunk {i+1}/{len(chunks)}...")
                try:
                    result = translator.translate(chunk, src=detected_lang, dest='en')
                    translated_chunks.append(result.text)
                except Exception as e:
                    print(f"[Translation] Error translating chunk {i+1}: {e}")
                    translated_chunks.append(chunk)
            
            translated_content = '\n'.join(translated_chunks)
            print(f"[Translation] Successfully translated {len(chunks)} chunks with googletrans")
            
            return translated_content, True
            
        except ImportError:
            print("[Translation] No translation libraries installed")
            print("[Translation] Install with: pip install deep-translator langdetect")
            print("[Translation] OR: pip install googletrans==4.0.0rc1 langdetect")
            return content, False
    
    except Exception as e:
        print(f"[Translation] Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return content, False


def _fetch_html_with_playwright(url: str) -> str:
    """
    Fetch fully-rendered HTML using Playwright (for JS-heavy pages).
    Returns empty string if Playwright isn't available or fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("[Playwright] Not installed - JS rendering disabled")
        print("[Playwright] Install with: pip install playwright")
        print("[Playwright] Then run: python -m playwright install chromium")
        return ""
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )
            page.goto(url, wait_until="networkidle", timeout=PLAYWRIGHT_TIMEOUT_MS)
            html = page.content()
            browser.close()
            return html or ""
    except Exception as e:
        print(f"[Playwright] Failed to render {url}: {e}")
        return ""


def _extract_with_bs4_html(html: str, url: str, translate_to_english: bool) -> Dict[str, str]:
    """
    Extract content from HTML using BeautifulSoup with aggressive cleaning.
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract publication logo
    publication_logo = _extract_publication_logo(soup, url)
    footer_logo = _extract_footer_logo(soup, url)
    font_family = _detect_article_font(soup)
    
    # Extract title
    title_tag = soup.find('title') or soup.find('h1')
    title = title_tag.get_text().strip() if title_tag else "Untitled"
    
    # Remove scripts, styles, navigation, ads, etc.
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'header']):
        tag.decompose()
    
    # Remove common junk classes/IDs
    junk_selectors = [
        {'class': ['nav', 'navigation', 'navbar', 'menu', 'sidebar', 'widget']},
        {'class': ['breadcrumb', 'breadcrumbs', 'tags', 'categories']},
        {'class': ['share', 'social', 'comments', 'related']},
        {'class': ['ad', 'ads', 'advertisement', 'promo']},
        {'class': ['meta', 'metadata', 'byline']},
        {'id': ['nav', 'navigation', 'sidebar', 'footer', 'header']},
    ]
    
    for selector in junk_selectors:
        for tag in soup.find_all(**selector):
            tag.decompose()
    
    # Get main content
    main_content = (
        soup.find('article') or 
        soup.find('main') or 
        soup.find('div', class_=['content', 'article', 'post', 'entry-content']) or
        soup.find('body')
    )
    
    if main_content:
        # Extract images with captions
        images = _extract_images_with_captions(main_content, url, limit=2)
        
        # Extract paragraphs for proper structure
        paragraphs = main_content.find_all('p')
        if paragraphs:
            content_parts = []
            
            # Add first image
            if images:
                img_html = f'<img src="{images[0]["src"]}" alt="Article image">'
                if images[0].get('caption'):
                    img_html += f'\n<figcaption>{images[0]["caption"]}</figcaption>'
                content_parts.append(img_html)
            
            # Add paragraphs
            for p in paragraphs:
                p_text = p.get_text().strip()
                if p_text:
                    content_parts.append(p_text)
            
            content = '\n\n'.join(content_parts)
        else:
            content = main_content.get_text(separator='\n\n').strip()
    else:
        content = ""
    
    # Clean up: remove multiple blank lines
    import re
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Translate if needed
    if translate_to_english and content:
        content, was_translated = _detect_and_translate_content(content, str(soup))
        if was_translated:
            print(f"[Translation] Content translated to English")
    
    return {
        "title": title,
        "author": "",
        "date": "",
        "content": content,
        "url": url,
        "publication_logo": publication_logo,
        "footer_logo": footer_logo,
        "font_family": font_family,
        "raw_html": str(soup)
    }


def fetch_webpage_content(url: str, translate_to_english: bool = True) -> Dict[str, str]:
    """
    Fetch and extract clean content from a webpage.
    
    Args:
        url: The URL to fetch
        
    Returns:
        {
            "title": "Article Title",
            "author": "Author Name",
            "date": "Publication Date",
            "content": "Full article text with <img> tags for images",
            "url": "Original URL",
            "publication_logo": "URL to publication logo/masthead (if found)",
            "footer_logo": "URL to footer logo (if found)",
            "font_family": "Detected font family from article",
            "raw_html": "Full HTML (for debugging)"
        }
    """
    # Handle direct PDF URLs by extracting text and re-wrapping
    pdf_result = _try_fetch_pdf_content(url, translate_to_english=translate_to_english)
    if pdf_result:
        return pdf_result

    # Try newspaper3k first (best for news/article sites)
    try:
        from newspaper import Article
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        
        article = Article(url)
        article.download()
        article.parse()
        
        # Get HTML and extract paragraphs + images manually for better formatting
        soup = BeautifulSoup(article.html, 'html.parser')
        
        # Extract publication logo/masthead for branding
        publication_logo = _extract_publication_logo(soup, url)
        
        # Extract footer logo
        footer_logo = _extract_footer_logo(soup, url)
        
        # Detect font family from article
        font_family = _detect_article_font(soup)
        
        # Find article body
        main_content = (
            soup.find('article') or 
            soup.find('main') or 
            soup.find('div', class_=['content', 'article', 'post', 'entry-content', 'article-body']) or
            soup.find('body')
        )
        
        if main_content:
            # Extract images with captions (filter out junk)
            images = _extract_images_with_captions(main_content, url, limit=2)
            
            # Extract paragraphs maintaining structure
            paragraphs = main_content.find_all('p')
            if paragraphs and len(paragraphs) > 3:
                # Build content with images interspersed
                content_parts = []
                
                # Add first image at top if available
                if images:
                    img_html = f'<img src="{images[0]["src"]}" alt="Article image">'
                    if images[0].get('caption'):
                        img_html += f'\n<figcaption>{images[0]["caption"]}</figcaption>'
                    content_parts.append(img_html)
                
                # Add paragraphs
                for p in paragraphs:
                    p_text = p.get_text().strip()
                    if p_text:
                        content_parts.append(p_text)
                
                # Add second image in middle if available
                if len(images) > 1 and len(content_parts) > 3:
                    mid_point = len(content_parts) // 2
                    img_html = f'<img src="{images[1]["src"]}" alt="Article image">'
                    if images[1].get('caption'):
                        img_html += f'\n<figcaption>{images[1]["caption"]}</figcaption>'
                    content_parts.insert(mid_point, img_html)
                
                content = '\n\n'.join(content_parts)
            else:
                # Fallback to newspaper3k text
                content = article.text or ""
                # Add main image if found
                if images:
                    img_html = f'<img src="{images[0]["src"]}" alt="Article image">'
                    if images[0].get('caption'):
                        img_html += f'\n<figcaption>{images[0]["caption"]}</figcaption>'
                    content = img_html + '\n\n' + content
        else:
            content = article.text or ""
        
        # Translate if needed
        if translate_to_english and content:
            content, was_translated = _detect_and_translate_content(content, article.html)
            if was_translated:
                print(f"[Translation] Content translated to English")

        # If content is thin, try JS-rendered HTML via Playwright
        if not content or len(content.strip()) < MIN_CONTENT_CHARS:
            html = _fetch_html_with_playwright(url)
            if html:
                return _extract_with_bs4_html(html, url, translate_to_english)
        
        return {
            "title": article.title or "Untitled",
            "author": ", ".join(article.authors) if article.authors else "",
            "date": article.publish_date.strftime("%B %d, %Y") if article.publish_date else "",
            "content": content,
            "url": url,
            "publication_logo": publication_logo,
            "footer_logo": footer_logo,
            "font_family": font_family,
            "raw_html": article.html
        }
    except Exception as e:
        print(f"[newspaper3k failed] {e}, trying BeautifulSoup...")
        # Fallback to BeautifulSoup with aggressive cleaning
        try:
            import requests
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; O1VisaBot/1.0)'
            })
            result = _extract_with_bs4_html(response.content, url, translate_to_english)
            
            # If content is thin, try JS-rendered HTML via Playwright
            if not result.get("content") or len(result["content"].strip()) < MIN_CONTENT_CHARS:
                html = _fetch_html_with_playwright(url)
                if html:
                    return _extract_with_bs4_html(html, url, translate_to_english)
            
            return result
        except Exception as e2:
            raise RuntimeError(f"Failed to fetch {url}: {e2}")


def _try_fetch_pdf_content(url: str, translate_to_english: bool = True) -> Optional[Dict[str, str]]:
    """
    Detect a PDF URL and extract text for consistent PDF output.
    
    Returns None if the URL does not appear to be a PDF.
    """
    url_lower = url.lower()
    is_pdf_url = url_lower.endswith(".pdf")
    
    try:
        import requests
        
        # Quick content-type check for non-.pdf URLs
        if not is_pdf_url:
            head = requests.head(url, timeout=10, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; O1VisaBot/1.0)'
            })
            content_type = head.headers.get("Content-Type", "").lower()
            if "application/pdf" not in content_type:
                return None
        
        # Download PDF bytes
        response = requests.get(url, timeout=20, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; O1VisaBot/1.0)'
        })
        response.raise_for_status()
        pdf_bytes = response.content
        
        # Extract text from PDF
        from src.pdf_text import extract_text_from_pdf_bytes
        content = extract_text_from_pdf_bytes(pdf_bytes) or ""
        
        # Translate if needed
        if translate_to_english and content:
            content, was_translated = _detect_and_translate_content(content, "")
            if was_translated:
                print(f"[Translation] PDF content translated to English")
        
        # Title from filename or URL
        filename = os.path.basename(url.split("?")[0])
        title = filename if filename else "Untitled"
        
        return {
            "title": title,
            "author": "",
            "date": "",
            "content": content,
            "url": url,
            "publication_logo": None,
            "footer_logo": None,
            "font_family": "Arial, Helvetica, sans-serif",
            "raw_html": ""
        }
    except Exception as e:
        print(f"[PDF fetch failed] {e}")
        return None

def _extract_publication_logo(soup, url: str) -> Optional[str]:
    """
    DISABLED: Logo extraction disabled - using text headers instead.
    
    This function now always returns None to force text fallback.
    Text headers look cleaner and more professional than image logos.
    
    Args:
        soup: BeautifulSoup object
        url: Original URL (for converting relative paths)
        
    Returns:
        None (always use text header)
    """
    # Always return None to use clean text header
    return None


def _extract_footer_logo(soup, url: str) -> Optional[str]:
    """
    DISABLED: Footer logo extraction disabled.
    
    Returns None to keep footer clean and text-only.
    
    Args:
        soup: BeautifulSoup object
        url: Original URL (for converting relative paths)
        
    Returns:
        None (always use text-only footer)
    """
    # Always return None for clean text footer
    return None


def _detect_article_font(soup) -> str:
    """
    Detect the font family used in the article from HTML/CSS.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Font family name (fallback to Arial if not detected)
    """
    font_family = "Arial, Helvetica, sans-serif"  # Default fallback
    
    # Method 1: Check article/main content for inline styles
    main_content = (
        soup.find('article') or 
        soup.find('main') or 
        soup.find('div', class_=['content', 'article', 'post'])
    )
    
    if main_content:
        style = main_content.get('style', '')
        if 'font-family' in style:
            # Extract font-family from inline style
            import re
            match = re.search(r'font-family:\s*([^;]+)', style)
            if match:
                font_family = match.group(1).strip()
    
    # Method 2: Check <style> tags for common article classes
    if font_family == "Arial, Helvetica, sans-serif":
        style_tags = soup.find_all('style')
        for style_tag in style_tags:
            style_content = style_tag.string or ""
            # Look for article/body font definitions
            if 'font-family' in style_content:
                import re
                # Try to find article or body font
                match = re.search(r'(?:article|\.article|body|\.content)[^}]*font-family:\s*([^;]+)', style_content)
                if match:
                    font_family = match.group(1).strip()
                    break
    
    # Clean up font family (remove quotes, clean syntax)
    font_family = font_family.replace('"', '').replace("'", "").strip()
    
    # Ensure we always have a fallback
    if not font_family or font_family == "":
        font_family = "Arial, Helvetica, sans-serif"
    
    return font_family


def _extract_images_with_captions(soup, url: str, limit: int = 2) -> List[Dict[str, str]]:
    """
    Extract editorial images with their captions/credits.
    Filters out UI chrome, ads, social widgets, etc.
    
    Args:
        soup: BeautifulSoup object (article body)
        url: Original URL (for converting relative paths)
        limit: Maximum number of images to extract
        
    Returns:
        [
            {"src": "https://...", "caption": "Photo credit"},
            ...
        ]
    """
    from urllib.parse import urljoin
    
    images = []
    
    # Comprehensive junk image filter
    junk_patterns = [
        # Logos/branding
        'logo', 'icon', 'brand', 'masthead',
        
        # Ads
        'ad', 'banner', 'sponsor', 'promo',
        
        # UI elements
        'button', 'nav', 'menu', 'header', 'footer', 'sidebar',
        'arrow', 'chevron', 'caret', 'hamburger',
        
        # Social
        'facebook', 'twitter', 'instagram', 'linkedin', 'social', 'share',
        
        # User elements
        'avatar', 'profile', 'user', 'author-photo',
        
        # Junk
        'pixel', 'tracking', 'beacon', 'analytics', 'widget',
        'thumbnail', 'badge', 'tag',
        
        # Subscription widgets
        'newsletter', 'subscribe', 'donate', 'support',
        
        # Placeholder/loading
        'placeholder', 'loading', 'spinner', 'loader'
    ]
    
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if not src:
            continue
        
        # Convert relative URLs to absolute
        img_src = urljoin(url, src)
        src_lower = img_src.lower()
        
        # Filter out junk images by URL
        if any(pattern in src_lower for pattern in junk_patterns):
            continue
        
        # Filter out junk images by CSS class
        img_classes = ' '.join(img.get('class', [])).lower()
        if any(pattern in img_classes for pattern in junk_patterns):
            continue
        
        # Skip tiny images (raised from 50px to 100px minimum)
        width = img.get('width')
        height = img.get('height')
        if width and height:
            try:
                if int(width) < 100 or int(height) < 100:
                    continue
            except (ValueError, TypeError):
                pass
        
        # Check if this is editorial content (not UI chrome)
        if not _is_editorial_image(img):
            continue
        
        # Extract caption (multiple methods)
        caption = _extract_image_caption(img)
        
        images.append({
            'src': img_src,
            'caption': caption
        })
        
        if len(images) >= limit:
            break
    
    return images


def _is_editorial_image(img) -> bool:
    """
    Distinguish editorial images (keep) from UI chrome (remove).
    
    Args:
        img: BeautifulSoup img tag
        
    Returns:
        True if this appears to be editorial content
    """
    src = img.get('src', '').lower()
    
    # Definite editorial content indicators
    editorial_indicators = [
        'photo', 'image', 'picture', 'gallery', 'media',
        'album', 'cover', 'artist', 'performer', 'concert'
    ]
    if any(ind in src for ind in editorial_indicators):
        return True
    
    # Check if inside article content areas
    parent = img.find_parent()
    if parent:
        parent_classes = ' '.join(parent.get('class', [])).lower()
        content_indicators = ['article', 'content', 'body', 'post', 'entry', 'main']
        if any(ind in parent_classes for ind in content_indicators):
            return True
    
    # Check if in header/footer/nav (definitely NOT editorial)
    chrome_parents = ['header', 'footer', 'nav', 'aside', 'sidebar']
    if any(img.find_parent(tag) for tag in chrome_parents):
        return False
    
    return True  # Default: keep it


def _extract_image_caption(img) -> Optional[str]:
    """
    Extract caption/credit for an image.
    
    Args:
        img: BeautifulSoup img tag
        
    Returns:
        Caption text, or None if not found
    """
    caption = None
    
    # Method 1: <figcaption> inside parent <figure>
    fig = img.find_parent('figure')
    if fig:
        figcaption = fig.find('figcaption')
        if figcaption:
            caption = figcaption.get_text(strip=True)
    
    # Method 2: <p class="caption"> nearby
    if not caption:
        caption_elem = img.find_next('p', class_=lambda x: x and 'caption' in str(x).lower())
        if caption_elem:
            caption = caption_elem.get_text(strip=True)
    
    # Method 3: <div class="credit"> or similar
    if not caption:
        credit_elem = img.find_next(['div', 'span'], class_=lambda x: x and (
            'credit' in str(x).lower() or 
            'photo-credit' in str(x).lower()
        ))
        if credit_elem:
            caption = credit_elem.get_text(strip=True)
    
    # Method 4: title or alt attribute
    if not caption:
        caption = img.get('title') or img.get('alt')
        # Skip generic alt text
        if caption and caption.lower() in ['image', 'photo', 'picture']:
            caption = None
    
    return caption


def convert_webpage_to_pdf_with_margins(
    webpage_data: Dict[str, str],
    left_margin_mm: float = 35,
    right_margin_mm: float = 35,
    top_margin_mm: float = 30,
    bottom_margin_mm: float = 30
) -> bytes:
    """
    Convert webpage content to PDF with authentic publication styling.
    Uses 30mm margins for annotation space.
    
    REQUIRES WeasyPrint - will fail if not installed.
    
    Args:
        webpage_data: Dictionary from fetch_webpage_content()
        left_margin_mm: Left margin in millimeters
        right_margin_mm: Right margin in millimeters
        top_margin_mm: Top margin in millimeters
        bottom_margin_mm: Bottom margin in millimeters
        
    Returns:
        PDF bytes
    """
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    
    # Extract data
    title = webpage_data.get('title', 'Untitled')
    author = webpage_data.get('author', '')
    date = webpage_data.get('date', '')
    url = webpage_data.get('url', '')
    content = webpage_data.get('content', '')
    publication_logo = webpage_data.get('publication_logo')
    footer_logo = webpage_data.get('footer_logo')
    font_family = webpage_data.get('font_family', 'Arial, Helvetica, sans-serif')
    
    # Get current timestamp for footer (like browser print)
    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M")
    
    # Shorten URL for display (remove https://)
    display_url = url.replace('https://', '').replace('http://', '')
    
    # Extract publication name from URL
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    publication_name = parsed_url.netloc.replace('www.', '').split('.')[0].title()
    
    # Build publication header HTML (LEFT-ALIGNED)
    publication_header = ''
    if publication_logo:
        publication_header = f'<div class="publication-header"><img src="{publication_logo}" class="publication-logo" alt="{publication_name}"></div>'
    else:
        publication_header = f'<div class="publication">{publication_name}</div>'
    
    # Build footer HTML (with footer logo if available)
    footer_content = f"""
        <div class="footer">
            © {datetime.now().year} {publication_name}. All rights reserved.<br>
            Original article: {display_url}<br>
            Retrieved: {timestamp}
        </div>
    """
    
    if footer_logo:
        footer_content = f"""
        <div class="footer">
            <img src="{footer_logo}" class="footer-logo" alt="{publication_name}"><br>
            © {datetime.now().year} {publication_name}. All rights reserved.<br>
            Original article: {display_url}<br>
            Retrieved: {timestamp}
        </div>
        """
    
    # Create HTML with authentic article styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: letter;
                margin: {top_margin_mm}mm {right_margin_mm}mm {bottom_margin_mm}mm {left_margin_mm}mm;
                
                @bottom-left {{
                    content: "{timestamp}";
                    font-family: {font_family};
                    font-size: 8pt;
                    color: #666;
                }}
                
                @bottom-center {{
                    content: "{publication_name}";
                    font-family: {font_family};
                    font-size: 8pt;
                    color: #666;
                    text-transform: uppercase;
                }}
                
                @bottom-right {{
                    content: counter(page) "/" counter(pages);
                    font-family: {font_family};
                    font-size: 8pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: {font_family};
                font-size: 10pt;
                line-height: 1.5;
                color: #333;
                text-align: justify;
                hyphens: none;
                margin: 0;
                padding: 0;
            }}
            
            .publication-header {{
                margin: 0 0 10pt 0;
                text-align: left;
            }}
            
            .publication-logo {{
                max-width: 200px;
                height: auto;
                display: block;
            }}
            
            .publication {{
                font-size: 9pt;
                color: #999;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 0 0 10pt 0;
                font-weight: bold;
                text-align: left;
            }}
            
            h1 {{
                font-size: 18pt;
                font-weight: bold;
                margin: 0 0 12pt 0;
                padding: 0;
                color: #000;
                line-height: 1.2;
                text-align: left;
            }}
            
            .byline {{
                font-size: 9pt;
                color: #666;
                margin: 0 0 8pt 0;
                font-style: italic;
                text-align: left;
            }}
            
            .divider {{
                border-bottom: 1px solid #ddd;
                margin: 8pt 0 6pt 0;
            }}
            
            .url-display {{
                font-size: 8pt;
                color: #999;
                margin: 0 0 6pt 0;
                text-decoration: none;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                text-align: left;
            }}
            
            .url-display::before {{
                content: "🌐 ";
                font-size: 10pt;
            }}
            
            .divider-bottom {{
                border-bottom: 1px solid #ddd;
                margin: 6pt 0 15pt 0;
            }}
            
            p {{
                margin: 0 0 12pt 0;
                padding: 0;
                text-indent: 0;
                orphans: 2;
                widows: 2;
            }}
            
            em, i {{
                font-style: italic;
            }}
            
            strong, b {{
                font-weight: bold;
            }}
            
            img {{
                max-width: 100%;
                max-height: 400pt;
                height: auto;
                display: block;
                margin: 20pt auto;
                border: 1px solid #eee;
                padding: 5pt;
            }}
            
            figcaption {{
                font-size: 8pt;
                color: #666;
                font-style: italic;
                text-align: center;
                margin: 5pt 0 15pt 0;
            }}
            
            .footer {{
                margin-top: 30pt;
                padding-top: 15pt;
                border-top: 1px solid #ddd;
                font-size: 8pt;
                color: #999;
                line-height: 1.4;
                text-align: left;
            }}
            
            .footer-logo {{
                max-width: 100px;
                height: auto;
                margin-bottom: 8pt;
            }}
        </style>
    </head>
    <body>
        {publication_header}
        
        <h1>{title}</h1>
        
        {f'<div class="byline">By {author}{", " + date if date else ""}</div>' if author or date else ''}
        
        <div class="divider"></div>
        
        <div class="url-display">{display_url}</div>
        
        <div class="divider-bottom"></div>
        
        <div class="content">
            {_format_content_to_html(content)}
        </div>
        
        {footer_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    font_config = FontConfiguration()
    html = HTML(string=html_template)
    pdf_bytes = html.write_pdf(font_config=font_config)
    
    return pdf_bytes


def _format_content_to_html(text: str) -> str:
    """
    Convert plain text to HTML paragraphs.
    Handles embedded <img> and <figcaption> tags.
    """
    if not text:
        return ""
    
    # Split into paragraphs
    parts = text.split('\n\n')
    
    html_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if this is an image tag
        if part.startswith('<img'):
            # Already HTML, keep as-is
            html_parts.append(part)
        elif part.startswith('<figcaption'):
            # Caption tag, keep as-is
            html_parts.append(part)
        else:
            # Text paragraph - escape HTML
            part = part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f"<p>{part}</p>")
    
    return '\n'.join(html_parts)


def batch_convert_urls_to_pdfs(
    urls_by_criterion: Dict[str, list],
    progress_callback=None,
    translate_to_english: bool = True
) -> Dict[str, Dict[str, bytes]]:
    """
    Convert multiple approved URLs to PDFs, organized by criterion.
    
    Args:
        urls_by_criterion: {"1": [{"url": "...", "title": "..."}], ...}
        progress_callback: Optional function to call with progress updates
        translate_to_english: If True, automatically translate non-English content
        
    Returns:
        {
            "1": {
                "Title_1.pdf": pdf_bytes,
                "Title_2.pdf": pdf_bytes
            },
            "3": {...}
        }
    """
    result = {}
    errors = []  # Collect errors to show later
    total_urls = sum(len(urls) for urls in urls_by_criterion.values())
    processed = 0
    
    for criterion_id, urls in urls_by_criterion.items():
        result[criterion_id] = {}
        
        for url_data in urls:
            url = url_data.get('url')
            title = url_data.get('title', 'Untitled')
            custom_filename = url_data.get('filename')  # Allow custom filename
            
            try:
                if progress_callback:
                    progress_callback(processed, total_urls, f"Fetching: {title}")
                
                # Fetch webpage (with translation if enabled)
                webpage_data = fetch_webpage_content(url, translate_to_english=translate_to_english)
                
                if progress_callback:
                    progress_callback(processed, total_urls, f"Converting: {title}")
                
                # Convert to PDF with slightly wider margins
                pdf_bytes = convert_webpage_to_pdf_with_margins(
                    webpage_data,
                    left_margin_mm=35,
                    right_margin_mm=35,
                    top_margin_mm=30,
                    bottom_margin_mm=30
                )
                
                # Use custom filename if provided, otherwise create safe filename from title
                if custom_filename:
                    filename = custom_filename
                else:
                    # Create safe filename
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
                    filename = f"{safe_title}.pdf"
                
                result[criterion_id][filename] = pdf_bytes
                processed += 1
                
            except Exception as e:
                error_msg = f"❌ {title}: {str(e)}"
                errors.append(error_msg)
                if progress_callback:
                    progress_callback(processed, total_urls, error_msg)
                processed += 1
                continue
    
    # Print errors so they show in Streamlit
    if errors:
        print("\n=== CONVERSION ERRORS ===")
        for err in errors:
            print(err)
        print("========================\n")
    
    return result
