import io
import fitz  # PyMuPDF


def _ocr_page(page) -> str:
    """
    OCR a single PDF page using PyMuPDF rendering + pytesseract.
    Returns empty string if OCR dependencies are unavailable.
    """
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        if not getattr(_ocr_page, "_warned_missing_deps", False):
            print("[OCR] pytesseract or Pillow not installed - OCR disabled")
            print("[OCR] Install with: pip install pytesseract pillow")
            print("[OCR] Also ensure Tesseract OCR is installed on the system")
            _ocr_page._warned_missing_deps = True
        return ""
    
    # Render at higher resolution for better OCR accuracy
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        if not getattr(_ocr_page, "_warned_missing_tesseract", False):
            print("[OCR] Tesseract OCR not available - OCR disabled")
            print("[OCR] Install Tesseract and ensure it's on PATH")
            _ocr_page._warned_missing_tesseract = True
        return ""


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    enable_ocr: bool = True,
    ocr_min_chars: int = 50
) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts = []
        for page in doc:
            page_text = page.get_text("text") or ""
            
            # OCR fallback for scanned pages with little/no text
            if enable_ocr and len(page_text.strip()) < ocr_min_chars:
                ocr_text = _ocr_page(page)
                if ocr_text.strip():
                    page_text = ocr_text
            
            parts.append(page_text)
        return "\n".join(parts)
    finally:
        doc.close()
