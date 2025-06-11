import os
import tempfile
import re
import logging

import streamlit as st
import pdfplumber
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
from PIL import Image

# ─── 1) SUPPRESS DEBUG LOGGING ───────────────────────────────────────────────
for lg in ("ppocr", "paddleocr", "paddle"):
    logging.getLogger(lg).setLevel(logging.ERROR)

# ─── 2) CACHE THE OCR MODEL ─────────────────────────────────────────────────
@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

ocr_model = load_ocr_model()

# ─── 3) CACHE YOUR OCR FUNCTION ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_part_numbers_file(path, min_length=2, conf_threshold=0.3):
    raw = ocr_model.ocr(path, cls=True)
    candidates = []
    for entry in raw:
        # Find the (text, score) tuple in whatever the entry returns
        text_info = next((e for e in entry if isinstance(e, tuple) and len(e) == 2), None)
        if not text_info:
            continue
        txt, score = text_info
        if score < conf_threshold:
            continue
        txt2 = txt.upper().replace(" ", "")
        if len(txt2) >= min_length and any(ch.isdigit() for ch in txt2):
            candidates.append((txt2, score))

    seen = set()
    out = []
    for txt2, sc in sorted(candidates, key=lambda x: -x[1]):
        if txt2 not in seen:
            seen.add(txt2)
            out.append((txt2, sc))
    return out


# ─── YOUR UI ─────────────────────────────────────────────────────────────────
st.title("MPN Matcher — Web Edition")

# 1) Invoice PDF uploader
pdf_file = st.file_uploader("Upload invoice PDF", type="pdf")
invoice_lines = []
if pdf_file:
    invoice_lines = extract_all_invoice_info_bytes(pdf_file.read())
    if invoice_lines:
        st.table(invoice_lines)
    else:
        st.error("No invoice lines found.")

# 2) Manual override or select first MPN
mpn_override = st.text_input("Or manual MPN override", value="")
mpn = mpn_override.strip() or (invoice_lines[0]["MPN"] if invoice_lines else "")

# 3) Photo uploader & matching (fixed-grid with placeholders)
imgs = st.file_uploader("Upload photos", type=["png","jpg","jpeg","bmp"], accept_multiple_files=True)
if imgs and mpn:
    st.write("### Matching Results")
    per_row = 4
    results = []

    # Show a spinner while this runs
    with st.spinner("🔍 Running OCR on your photos…"):
        for img in imgs:
            path = img.name
            with open(path, "wb") as f:
                f.write(img.getbuffer())

            try:
                best = 0
                for txt, score in extract_part_numbers_file(path):
                    best = max(best, score)

            except Exception as e:
                st.error(f"❌ OCR failed on {img.name}: {e}")
                best = 0

            # Colour-code
            if best == 100:
                bg, fg = "#28a745", "white"
            elif best >= 85:
                bg, fg = "white", "black"
            else:
                bg, fg = "#dc3545", "white"

            results.append({
                "name": img.name,
                "path": path,
                "score": best,
                "bg": bg,
                "fg": fg,
            })

    # Layout in a fixed 4-column grid
    for i in range(0, len(results), per_row):
        chunk = results[i : i + per_row]
        cols = st.columns(per_row)
        for idx, col in enumerate(cols):
            with col:
                if idx < len(chunk):
                    item = chunk[idx]
                    thumb = Image.open(item["path"])
                    thumb.thumbnail((150, 150))
                    st.image(thumb, width=150, caption=item["name"])
                    st.markdown(
                        f"<div style='background:{item['bg']};"
                        f"color:{item['fg']};padding:4px;"
                        f"border-radius:4px;text-align:center;'>"
                        f"**{item['score']:.1f}%**"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("")  # keep the column width consistent

            # cleanup disk after rendering
            if idx < len(chunk):
                os.remove(chunk[idx]["path"])
