import os
import re
import logging
import pdfplumber
import numpy as np
np.int = int  # PaddleOCR’s NumPy monkey‐patch
import cv2
from PIL import Image, ImageTk
from difflib import SequenceMatcher
from paddleocr import PaddleOCR

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
# Suppress PaddleOCR’s verbose startup logs
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)

# ─── Invoice‐Parsing Functions ──────────────────────────────────────────────────
def get_pdf_full_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        messagebox.showerror("PDF Error", f"Could not read PDF:\n{e}")
        return ""

def parse_po_from_date_line(full_text):
    for line in full_text.splitlines():
        m = re.match(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{4}\s+([A-Za-z0-9-]+)", line)
        if m:
            return m.group(1)
    m2 = re.search(r"PO#\s*([A-Za-z0-9-]+)", full_text, re.IGNORECASE)
    return m2.group(1) if m2 else ""

def extract_ship_to_block(pdf_path: str) -> str:
    text = get_pdf_full_text(pdf_path)
    if not text:
        return ""
    m = re.search(r"Ship To:\s*([\s\S]*?)(?=\n\s*Customer #:)", text, re.IGNORECASE)
    raw = m.group(1).strip() if m else ""
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw = re.sub(r"\n\s*\n+", " ", raw)
    lines = [ln.strip().rstrip(",") for ln in raw.split('\n') if ln.strip()]
    return "\n".join(lines)

def extract_all_invoice_info(pdf_path):
    """
    Returns a list of invoice‐line dicts. Handles multiple Ref#s per PDF.
    """
    full_text = get_pdf_full_text(pdf_path)
    if not full_text:
        return []

    po   = parse_po_from_date_line(full_text)
    so   = (re.search(r"\b(\d{6}/\d{2})\b", full_text) or ["",""])[1]
    ship = extract_ship_to_block(pdf_path)

    lines = full_text.splitlines()
    raw_refs = re.findall(r'\b\d{6}-\d+(?:\.\d+)?\b', full_text)
    unique_refs = []
    for r in raw_refs:
        if r not in unique_refs:
            unique_refs.append(r)

    invoices = []
    for ref in unique_refs:
        try:
            idx = next(i for i, L in enumerate(lines) if ref in L)
        except StopIteration:
            continue
        start  = max(0, idx - 10)
        end    = min(len(lines), idx + 10)
        window = "\n".join(lines[start:end])

        qty_m = re.search(r"([\d,]+)\s+PCS", window)
        qty   = qty_m.group(1) if qty_m else ""

        mpn_m = re.search(r"^Manuf\. Part#\s*:\s*(\S+)", window, re.MULTILINE)
        mpn   = mpn_m.group(1) if mpn_m else ""

        mf_m  = re.search(r"^Manufacturer\s*:\s*(.+)$", window, re.MULTILINE)
        mf    = mf_m.group(1).strip() if mf_m else ""

        cust = "NA"
        cust_m = re.search(
            r"^Cust\. Part#\s*:\s*(\S+)$",
            window,
            re.MULTILINE | re.IGNORECASE
        )
        if cust_m:
            token = cust_m.group(1).strip()
            if token and " " not in token:
                cust = token

        invoices.append({
            "Purchase Order Number":         po,
            "SO Number":                     so,
            "ShipToAddress":                 ship,
            "RefNumber":                     ref,
            "Quantity":                      qty,
            "Manufacturer Part Number":      mpn,
            "Manufacturer":                  mf,
            "Customer Internal Part Number": cust
        })

    return invoices

# ─── OCR + Fuzzy‐Match ───────────────────────────────────────────────────────────
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def extract_part_numbers(image_path, min_length=2, conf_threshold=0.3):
    img = cv2.imread(image_path)
    if img is None:
        return []
    raw = ocr_model.ocr(image_path, cls=True)
    candidates = []
    for _, (txt, score) in raw:
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

def match_ratio(target: str, candidate: str) -> float:
    sm = SequenceMatcher(None, target, candidate)
    matched = sum(b.size for b in sm.get_matching_blocks()[:-1])
    return (matched / max(len(target), 1)) * 100

# ─── Main GUI ──────────────────────────────────────────────────────────────────
class MatcherGUI(tk.Tk):
    MAX_PHOTOS = 60          # Changed from 12 → 60
    THUMB_SIZE = (150, 150)  # Slightly larger thumbnails

    def __init__(self):
        super().__init__()
        self.title("Fusion Worldwide — Part# Matcher")
        # Fix window size and disable resizing
        self.geometry("700x1000")
        self.resizable(False, False)
        self.configure(bg="white")

        # Keep references to PhotoImage so they aren’t garbage‐collected
        self.thumbnails = [None] * self.MAX_PHOTOS

        # Invoice data:
        self.invoice_lines = []
        self.selected_line = None
        self.manual_mpn_var = tk.StringVar()

        # Photo data:
        self.photo_paths = []

        self._build_ui()

    def _build_ui(self):
        # ——— Header (14 pt) ———
        header = tk.Label(
            self,
            text="MPN Matcher",
            bg="white", fg="#333",
            font=("Helvetica Neue", 14, "bold")
        )
        header.pack(pady=(4, 2))

        # ─── Section 1: Invoice PDF & Line ─────────────────────────────────────
        inv_frame = tk.LabelFrame(
            self,
            text="1. Choosing invoice PDF & Line",
            bg="white", fg="#000",
            font=("Helvetica Neue", 11, "bold"),
            bd=1, relief="solid",
            padx=6, pady=6
        )
        inv_frame.pack(fill="x", padx=6, pady=4)

        # Invoice PDF label + Browse
        tk.Label(
            inv_frame,
            text="Invoice PDF:",
            bg="white",
            font=("Helvetica Neue", 9)
        ).grid(row=0, column=0, sticky="e", padx=4, pady=4)

        self.inv_lbl = tk.Label(
            inv_frame,
            text="No file selected",
            bg="white", fg="gray",
            font=("Helvetica Neue", 9)
        )
        self.inv_lbl.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        browse_btn = ttk.Button(inv_frame, text="Browse…", command=self.load_invoice)
        browse_btn.grid(row=0, column=2, padx=4, pady=4)

        clear_btn = ttk.Button(inv_frame, text="Clear All", command=self.clear_all)
        clear_btn.grid(row=0, column=3, padx=4, pady=4)

        # “Select Line” combobox
        tk.Label(
            inv_frame,
            text="Select Line:",
            bg="white",
            font=("Helvetica Neue", 9)
        ).grid(row=1, column=0, sticky="e", padx=4, pady=4)

        self.line_combo = ttk.Combobox(
            inv_frame,
            state="disabled",
            font=("Helvetica Neue", 9)
        )
        self.line_combo.grid(row=1, column=1, columnspan=3, sticky="we", padx=4, pady=4)
        self.line_combo.bind("<<ComboboxSelected>>", lambda e: self.on_line_selected())

        # Manual part‐number entry
        tk.Label(
            inv_frame,
            text="Or Manual enter MPN:",
            bg="white",
            font=("Helvetica Neue", 9, "italic")
        ).grid(row=2, column=0, sticky="e", padx=4, pady=4)

        tk.Entry(
            inv_frame,
            textvariable=self.manual_mpn_var,
            font=("Helvetica Neue", 9),
            bg="white", bd=1, relief="solid"
        ).grid(row=2, column=1, columnspan=3, sticky="we", padx=4, pady=4)

        inv_frame.columnconfigure(1, weight=1)

        # Thinner separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=6, pady=6)

        # ─── Section 2: Photos (20 slots in 5×4 grid, inside a scrollable area) ───
        photo_outer = tk.LabelFrame(
            self,
            text=f"2. Select photos - Maximum {self.MAX_PHOTOS}",
            bg="white", fg="#000",
            font=("Helvetica Neue", 11, "bold"),
            bd=1, relief="solid",
            padx=6, pady=6
        )
        photo_outer.pack(fill="both", expand=True, padx=6, pady=4)

        select_btn = ttk.Button(photo_outer, text="Select Photos…", command=self.load_photos)
        select_btn.pack(anchor="w", pady=(0, 6), padx=4)

        # ─── Create a scrollable playground for the 20 slots ──────────────────
        canvas = tk.Canvas(photo_outer, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(photo_outer, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        # Whenever scrollable_frame changes size, update canvas scrollregion
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # ─── ADD MOUSE‐WHEEL BINDING ───────────────────────────────────────────
        def _on_mousewheel(event):
            # On Windows, event.delta is a multiple of 120. 
            # Multiply by −1 because positive delta should scroll up.
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        # ─────────────────────────────────────────────────────────────────────────

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Now build the 5×4 grid INSIDE scrollable_frame
        self.photo_slots = []
        rows = 15
        cols = 4
        for r in range(rows):
            for c in range(cols):
                slot_frame = tk.Frame(scrollable_frame, bg="white")
                slot_frame.grid(row=r, column=c, padx=6, pady=6, sticky="n")

                canvas_slot = tk.Canvas(
                    slot_frame,
                    width=self.THUMB_SIZE[0],
                    height=self.THUMB_SIZE[1],
                    bg="#f0f0f0",
                    bd=1, relief="solid",
                    highlightthickness=0
                )
                canvas_slot.pack()

                fn_lbl = tk.Label(
                    slot_frame,
                    text="—",
                    bg="white", fg="gray",
                    font=("Helvetica Neue", 9),
                    wraplength=self.THUMB_SIZE[0]
                )
                fn_lbl.pack(pady=(4, 2))

                pct_lbl = tk.Label(
                    slot_frame,
                    text="—",
                    bg="white",
                    font=("Helvetica Neue", 10, "bold"),
                    padx=4, pady=2
                )
                pct_lbl.pack()

                self.photo_slots.append((canvas_slot, fn_lbl, pct_lbl))

        # Make columns expand equally (optional; mostly for consistent spacing)
        for c in range(cols):
            scrollable_frame.columnconfigure(c, weight=1)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=6, pady=6)
        self._clear_all_slots()

    def load_invoice(self):
        """
        Ask user to select a PDF. Then parse out all lines via extract_all_invoice_info().
        Populate self.invoice_lines and enable the combobox.
        """
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not path:
            return

        self.inv_lbl.config(text=os.path.basename(path), fg="black")
        self.invoice_lines = extract_all_invoice_info(path)
        if not self.invoice_lines:
            messagebox.showerror("Parse Error", "No reference lines found.")
            return

        options = [
            f"{d['RefNumber']}  ({d['Manufacturer Part Number'] or '—'})"
            for d in self.invoice_lines
        ]
        self.line_combo.config(values=options, state="readonly")
        if len(options) == 1:
            self.line_combo.current(0)
            self.on_line_selected()
        else:
            self.line_combo.set("Select…")

        # Clear manual part # and any photos
        self.manual_mpn_var.set("")
        self.photo_paths = []
        self._clear_all_slots()

    def on_line_selected(self):
        idx = self.line_combo.current()
        self.selected_line = (
            self.invoice_lines[idx] if (0 <= idx < len(self.invoice_lines)) else None
        )
        # Clear any previously‐loaded photos
        self.manual_mpn_var.set("")
        self.photo_paths = []
        self._clear_all_slots()

    def load_photos(self):
        """
        Ask user to pick up to MAX_PHOTOS. Then start the per-slot processing
        so that each iteration returns control to the event loop.
        """
        chosen_part = self.manual_mpn_var.get().strip() or \
                      (self.selected_line or {}).get("Manufacturer Part Number", "")
        if not chosen_part:
            messagebox.showwarning(
                "No part #",
                "Please select an invoice line OR enter a part number above."
            )
            return

        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not paths:
            return

        if len(paths) > self.MAX_PHOTOS:
            messagebox.showwarning(
                "Too Many Photos",
                f"Please select up to {self.MAX_PHOTOS} photos."
            )
            return

        # Store selected photo paths
        self.photo_paths = list(paths)

        # Clear any existing thumbnails/labels
        self._clear_all_slots()

        # Create overlay and start processing slots one by one
        self._start_processing_slots()

    def clear_all(self):
        """
        Clear everything in Section 1 and Section 2:
        - Reset invoice label, disable combobox, clear manual MPN
        - Clear any loaded photos/thumbnails
        """
        self.inv_lbl.config(text="No file selected", fg="gray")
        self.invoice_lines = []
        self.selected_line = None

        self.line_combo.set("")  # clear text
        self.line_combo.config(values=[], state="disabled")

        self.manual_mpn_var.set("")

        self.photo_paths = []
        self._clear_all_slots()

    def _clear_all_slots(self):
        for canvas_slot, fn_lbl, pct_lbl in self.photo_slots:
            canvas_slot.delete("all")
            canvas_slot.create_rectangle(
                0, 0, self.THUMB_SIZE[0], self.THUMB_SIZE[1],
                fill="#f0f0f0", outline="#bbb"
            )
            fn_lbl.config(text="—", fg="gray")
            pct_lbl.config(text="—", bg="white", fg="black")
        self.thumbnails = [None] * self.MAX_PHOTOS

    def _start_processing_slots(self):
        """
        Create a small “Processing…” overlay and initialize processing index,
        then schedule the first slot via after(...).
        """
        overlay = tk.Toplevel(self)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        self.update_idletasks()
        w, h = 140, 34
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (w // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (h // 2)
        overlay.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(
            overlay,
            text="Processing…",
            font=("Helvetica Neue", 10),
            bg="#ffffe0", fg="#333"
        ).pack(fill="both", expand=True)
        self.update_idletasks()

        self._processing_overlay = overlay
        self._processing_index = 0

        self.after(50, self._process_next_slot)

    def _process_next_slot(self):
        """
        Processes just one photo/slot, updates its thumbnail and match %,
        then schedules the next slot (if any) via after(...).
        """
        i = self._processing_index
        total = len(self.photo_paths)

        # Pull out references to the slot widgets
        canvas_slot, fn_lbl, pct_lbl = self.photo_slots[i]

        # 1) Update filename label
        img_path = self.photo_paths[i]
        name = os.path.basename(img_path)
        if len(name) > 18:
            name = name[:15] + "..."
        fn_lbl.config(text=name, fg="black")

        # 2) Create and center the thumbnail
        try:
            pil = Image.open(img_path)
            pil.thumbnail(self.THUMB_SIZE, Image.LANCZOS)
            thumb_bg = Image.new("RGB", self.THUMB_SIZE, "#f0f0f0")
            w_img, h_img = pil.size
            x_off = (self.THUMB_SIZE[0] - w_img) // 2
            y_off = (self.THUMB_SIZE[1] - h_img) // 2
            thumb_bg.paste(pil, (x_off, y_off))
            tk_img = ImageTk.PhotoImage(thumb_bg)
            self.thumbnails[i] = tk_img

            canvas_slot.delete("all")
            canvas_slot.create_image(
                self.THUMB_SIZE[0] // 2,
                self.THUMB_SIZE[1] // 2,
                image=tk_img
            )
        except Exception as e:
            logger.error(f"Error generating thumbnail for {img_path}: {e}")
            canvas_slot.delete("all")
            canvas_slot.create_rectangle(
                0, 0, self.THUMB_SIZE[0], self.THUMB_SIZE[1],
                fill="#f0f0f0", outline="#bbb"
            )

        # 3) Compute the best match percentage for this slot
        target_mpn = self.manual_mpn_var.get().strip() or \
                     (self.selected_line or {}).get("Manufacturer Part Number", "")

        best_pct = 0.0
        try:
            for txt, _ in extract_part_numbers(img_path):
                best_pct = max(best_pct, match_ratio(target_mpn, txt))
        except Exception as e:
            logger.error(f"OCR failed on {img_path}: {e}")
            best_pct = 0.0

        best_pct = round(best_pct, 1)
        pct_lbl.config(text=f"{best_pct:.1f}%", padx=4, pady=2)
        if best_pct >= 100.0:
            bg_color, fg_color = "#28a745", "white"
        elif best_pct >= 80.0:
            bg_color, fg_color = "white", "black"
        else:
            bg_color, fg_color = "#dc3545", "white"

        pct_lbl.config(bg=bg_color, fg=fg_color)

        # 4) Move on to the next slot (if any)
        self._processing_index += 1
        if self._processing_index < total:
            self.after(50, self._process_next_slot)
        else:
            # All chosen photos are done—destroy the overlay and clear any empty slots
            try:
                self._processing_overlay.destroy()
            except Exception:
                pass
            del self._processing_overlay

            # Fill in remaining slots (if fewer than MAX_PHOTOS) with placeholders
            for j in range(total, self.MAX_PHOTOS):
                c, fn, pl = self.photo_slots[j]
                c.delete("all")
                c.create_rectangle(
                    0, 0, self.THUMB_SIZE[0], self.THUMB_SIZE[1],
                    fill="#f0f0f0", outline="#bbb"
                )
                fn.config(text="—", fg="gray")
                pl.config(text="—", bg="white", fg="black")

            # Done. The GUI remains responsive throughout.

if __name__ == "__main__":
    MatcherGUI().mainloop()
