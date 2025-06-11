import numpy as np
# --- restore np.int so PaddleOCR’s legacy code won’t break ---
np.int = int

import os
import tempfile
import re

import streamlit as st
import pdfplumber
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
from PIL import Image

# … the rest of your code …



# ─── Helper functions ─────────────────────────────────────────────────────────

def get_pdf_full_text_bytes(b: bytes) -> str:
    """Extract all text from a PDF passed in as bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b)
        path = tmp.name
    try:
        text = "\n".join(page.extract_text() or "" for page in pdfplumber.open(path).pages)
    finally:
        os.remove(path)
    return text

def parse_po_from_date_line(full_text: str) -> str:
    for line in full_text.splitlines():
        m = re.match(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{4}\s+([A-Za-z0-9-]+)", line)
        if m:
            return m.group(1)
    m2 = re.search(r"PO#\s*([A-Za-z0-9-]+)", full_text, re.IGNORECASE)
    return m2.group(1) if m2 else ""

def extract_ship_to_block(full_text: str) -> str:
    m = re.search(r"Ship To:\s*([\s\S]*?)(?=\n\s*Customer #:)", full_text, re.IGNORECASE)
    raw = m.group(1).strip() if m else ""
    raw = raw.replace('\r\n','\n').replace('\r','\n')
    raw = re.sub(r"\n\s*\n+"," ", raw)
    lines = [ln.strip().rstrip(",") for ln in raw.split('\n') if ln.strip()]
    return "\n".join(lines)

def extract_all_invoice_info_bytes(b: bytes):
    """Return list of dicts for each invoice line in the uploaded PDF."""
    text = get_pdf_full_text_bytes(b)
    if not text:
        return []
    po = parse_po_from_date_line(text)
    so = (re.search(r"\b(\d{6}/\d{2})\b", text) or ["",""])[1]
    ship = extract_ship_to_block(text)

    lines = text.splitlines()
    raw_refs = re.findall(r'\b\d{6}-\d+(?:\.\d+)?\b', text)
    unique_refs = []
    for r in raw_refs:
        if r not in unique_refs:
            unique_refs.append(r)

    invoices = []
    for ref in unique_refs:
        idx = next((i for i, L in enumerate(lines) if ref in L), None)
        if idx is None:
            continue
        window = "\n".join(lines[max(0, idx-10): idx+10])
        qty   = (re.search(r"([\d,]+)\s+PCS", window) or ["",""])[1]
        mpn   = (re.search(r"^Manuf\. Part#\s*:\s*(\S+)", window, re.M) or ["",""])[1]
        mf    = (re.search(r"^Manufacturer\s*:\s*(.+)$", window, re.M) or ["",""])[1].strip()
        cust  = "NA"
        cm    = re.search(r"^Cust\. Part#\s*:\s*(\S+)$", window, re.M|re.I)
        if cm:
            token = cm.group(1).strip()
            if token and " " not in token:
                cust = token

        invoices.append({
            "RefNumber": ref,
            "MPN": mpn,
            "Manufacturer": mf,
            "Quantity": qty,
            "SO Number": so,
            "PO Number": po,
            "ShipTo": ship,
            "CustPart#": cust
        })
    return invoices

# ─── OCR + Fuzzy‐Match ───────────────────────────────────────────────────────────
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def extract_part_numbers_file(path, min_length=2, conf_threshold=0.3):
    raw = ocr_model.ocr(path, cls=True)
    candidates = []
    for entry in raw:
        # entry might look like [box_points, (txt,score)] or
        # [box_points, (txt,score), cls_info]. So find the tuple:
        text_info = next(
            (e for e in entry if isinstance(e, tuple) and len(e) == 2),
            None
        )
        if text_info is None:
            continue
        txt, score = text_info

        if score < conf_threshold:
            continue
        txt2 = txt.upper().replace(" ", "")
        if len(txt2) >= min_length and any(ch.isdigit() for ch in txt2):
            candidates.append((txt2, score))

    # remove duplicates, sort by score
    seen = set()
    out = []
    for txt2, sc in sorted(candidates, key=lambda x: -x[1]):
        if txt2 not in seen:
            seen.add(txt2)
            out.append((txt2, sc))
    return out


def match_ratio(a: str, b: str) -> float:
    sm = SequenceMatcher(None, a, b)
    matched = sum(block.size for block in sm.get_matching_blocks()[:-1])
    return (matched / max(len(a), 1)) * 100

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
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
imgs = st.file_uploader(
    "Upload photos",
    type=["png","jpg","jpeg","bmp"],
    accept_multiple_files=True
)
if imgs and mpn:
    st.write("### Matching Results")
    per_row = 4
    results = []

    # collect results as before…
    for img in imgs:
        path = img.name
        with open(path, "wb") as f:
            f.write(img.getbuffer())

        best = 0
        for txt, _ in extract_part_numbers_file(path):
            best = max(best, match_ratio(mpn, txt))

        if best == 100:
            bg, fg = "#28a745", "white"
        elif best >= 85:
            bg, fg = "white", "black"
        else:
            bg, fg = "#dc3545", "white"

        results.append({"name": img.name, "path": path, "score": best, "bg": bg, "fg": fg})

    # render in rows of exactly per_row columns
    for start in range(0, len(results), per_row):
        chunk = results[start:start+per_row]
        cols = st.columns(per_row)   # ALWAYS get 4 columns
        for i, col in enumerate(cols):
            if i < len(chunk):
                item = chunk[i]
                with col:
                    thumb = Image.open(item["path"])
                    thumb.thumbnail((150, 150))
                    st.image(thumb, width=150, caption=item["name"])
                    st.markdown(
                        f"<div style='background:{item['bg']};"
                        f"color:{item['fg']};padding:4px;border-radius:4px;"
                        f"text-align:center;'>"
                        f"**{item['score']:.1f}%**"
                        "</div>",
                        unsafe_allow_html=True
                    )
                os.remove(item["path"])
            else:
                # placeholder: leave blank so column stays the same width
                cols[i].write("")  
