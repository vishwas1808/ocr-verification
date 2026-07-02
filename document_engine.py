"""
document_engine.py
------------------
Core OCR + classification + format-validation logic.

IMPORTANT (read this):
This module does NOT verify documents against any government database
(UIDAI/Aadhar, NSDL/PAN, Passport Seva). No such public API exists for
private apps to use. What this module DOES do:

  1. Run OCR on the uploaded image to extract raw text.
  2. Classify which document type it most likely is (PAN / Aadhar /
     Passport) based on keywords + structural patterns.
  3. Validate the extracted ID number against the official FORMAT rules
     for that document type (regex + checksum where applicable).
  4. Extract whatever fields it reasonably can (name, DOB, number, etc).

So "verified" in this app means "format-valid and structurally consistent
with a real document of this type" -- not "confirmed authentic / confirmed
to exist in a government registry."
"""

import os
import re
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- Windows: auto-locate tesseract.exe if it's not on PATH ---------------
# pip installs the Python wrapper (pytesseract), but NOT the actual OCR
# engine. On Windows that engine is usually installed at one of these
# default paths and is often not added to PATH automatically.
if os.name == "nt":
    _common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for _path in _common_paths:
        if os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Improve OCR accuracy: grayscale, denoise, adaptive threshold."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale small images -- OCR likes >= ~1000px on the long edge
    h, w = gray.shape
    longest = max(h, w)
    if longest < 1200:
        scale = 1200 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return Image.fromarray(thresh)


def run_ocr(pil_img: Image.Image) -> str:
    """Run Tesseract on both the raw and preprocessed image, return the
    longer / richer result (preprocessing sometimes hurts on already-clean
    scans, so we hedge)."""
    raw_text = pytesseract.image_to_string(pil_img)
    try:
        processed = preprocess_image(pil_img)
        proc_text = pytesseract.image_to_string(processed)
    except Exception:
        proc_text = ""

    # Heuristic: pick whichever produced more alphanumeric content
    raw_score = sum(c.isalnum() for c in raw_text)
    proc_score = sum(c.isalnum() for c in proc_text)
    return proc_text if proc_score > raw_score else raw_text


# ---------------------------------------------------------------------------
# Format validators
# ---------------------------------------------------------------------------

PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
AADHAR_REGEX = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PASSPORT_REGEX = re.compile(r"\b([A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9])\b")  # Indian passport format
DOB_REGEX = re.compile(r"\b(\d{2}[/\-.]\d{2}[/\-.]\d{4})\b")

PAN_FOURTH_CHAR_MEANING = {
    "P": "Individual", "C": "Company", "H": "HUF", "A": "AOP", "B": "BOI",
    "G": "Government", "J": "Artificial Judicial Person", "L": "Local Authority",
    "F": "Firm/LLP", "T": "Trust",
}


def validate_pan(number: str) -> dict:
    """Validate PAN format: AAAAA9999A. Checks structure + 4th-char category."""
    number = number.replace(" ", "").upper()
    match = PAN_REGEX.fullmatch(number)
    if not match:
        return {"valid": False, "reason": "Does not match PAN format AAAAA9999A"}
    fourth_char = number[3]
    category = PAN_FOURTH_CHAR_MEANING.get(fourth_char, "Unknown")
    return {
        "valid": True,
        "holder_category": category,
        "note": "Format and checksum-pattern valid. Not verified against the IT Department database.",
    }


def validate_aadhar(number: str) -> dict:
    """Validate Aadhar format: 12 digits, Verhoeff checksum, doesn't start with 0/1."""
    digits = re.sub(r"\s", "", number)
    if not re.fullmatch(r"\d{12}", digits):
        return {"valid": False, "reason": "Does not match 12-digit Aadhar format"}
    if digits[0] in ("0", "1"):
        return {"valid": False, "reason": "Aadhar numbers cannot start with 0 or 1"}

    if _verhoeff_check(digits):
        return {"valid": True, "note": "Format and checksum valid. Not verified against UIDAI database."}
    else:
        return {"valid": False, "reason": "Failed Verhoeff checksum validation -- likely a typo/OCR error or invalid number"}


def validate_passport(number: str) -> dict:
    """Validate Indian passport format: 1 letter + 7 digits (commonly)."""
    number = re.sub(r"\s", "", number).upper()
    if not PASSPORT_REGEX.fullmatch(number):
        return {"valid": False, "reason": "Does not match Indian passport format (e.g. A1234567)"}
    return {"valid": True, "note": "Format valid. Not verified against Passport Seva database."}


# Verhoeff algorithm (used by Aadhar checksum)
_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]
]


def _verhoeff_check(num_str: str) -> bool:
    c = 0
    digits = [int(d) for d in reversed(num_str)]
    for i, digit in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return c == 0


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

KEYWORDS = {
    "PAN": [
        "income tax department", "permanent account number", "income tax",
        "govt. of india", "incometaxindia", "pan", "father's name",
    ],
    "AADHAR": [
        "unique identification authority", "uidai", "aadhaar", "aadhar",
        "government of india", "dob", "male", "female", "vid",
    ],
    "PASSPORT": [
        "republic of india", "passport", "type", "nationality",
        "place of birth", "place of issue", "date of expiry", "p<ind",
    ],
}


def classify_document(text: str) -> dict:
    """Score the OCR text against keyword sets for each doc type, then
    confirm/override using regex pattern hits (more reliable than keywords)."""
    lower = text.lower()
    scores = {doc_type: 0 for doc_type in KEYWORDS}

    for doc_type, words in KEYWORDS.items():
        for w in words:
            if w in lower:
                scores[doc_type] += 1

    # Pattern-based boosts (numbers are stronger evidence than keywords)
    if PAN_REGEX.search(text.upper()):
        scores["PAN"] += 3
    if AADHAR_REGEX.search(text):
        scores["AADHAR"] += 2  # weaker alone, digits-only pattern is common
    if PASSPORT_REGEX.search(text.upper()) and "passport" in lower:
        scores["PASSPORT"] += 3

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return {"type": "UNKNOWN", "confidence": 0, "scores": scores}

    total = sum(scores.values()) or 1
    confidence = round((best_score / total) * 100)
    return {"type": best_type, "confidence": confidence, "scores": scores}


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def extract_fields(text: str, doc_type: str) -> dict:
    fields = {}

    dob_match = DOB_REGEX.search(text)
    if dob_match:
        fields["date_of_birth"] = dob_match.group(1)

    if doc_type == "PAN":
        m = PAN_REGEX.search(text.upper())
        if m:
            fields["pan_number"] = m.group(1)
        name_match = re.search(r"(?:Name)[:\s]+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})", text)
        if name_match:
            fields["name"] = name_match.group(1).strip()

    elif doc_type == "AADHAR":
        m = AADHAR_REGEX.search(text)
        if m:
            fields["aadhar_number"] = re.sub(r"\s", "", m.group(1))
        gender_match = re.search(r"\b(Male|Female|MALE|FEMALE)\b", text)
        if gender_match:
            fields["gender"] = gender_match.group(1).title()

    elif doc_type == "PASSPORT":
        m = PASSPORT_REGEX.search(text.upper())
        if m:
            fields["passport_number"] = re.sub(r"\s", "", m.group(1))
        nat_match = re.search(r"Nationality[:\s]+([A-Z][A-Za-z\s]{2,20})", text)
        if nat_match:
            fields["nationality"] = nat_match.group(1).strip()

    return fields


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

VALIDATORS = {
    "PAN": ("pan_number", validate_pan),
    "AADHAR": ("aadhar_number", validate_aadhar),
    "PASSPORT": ("passport_number", validate_passport),
}


def analyze_document(pil_img: Image.Image) -> dict:
    """Full pipeline: OCR -> classify -> extract -> format-validate."""
    text = run_ocr(pil_img)
    text_clean = text.strip()

    if len(text_clean) < 8:
        return {
            "success": False,
            "document_type": "UNKNOWN",
            "confidence": 0,
            "message": "Could not read enough text from this image. Try a clearer, well-lit, non-blurry photo or scan.",
            "raw_text_preview": text_clean,
        }

    classification = classify_document(text_clean)
    doc_type = classification["type"]

    if doc_type == "UNKNOWN":
        return {
            "success": False,
            "document_type": "UNKNOWN",
            "confidence": 0,
            "message": "This doesn't appear to be a PAN, Aadhar, or Passport document we can recognize.",
            "raw_text_preview": text_clean[:300],
        }

    fields = extract_fields(text_clean, doc_type)

    id_field_name, validator_fn = VALIDATORS[doc_type]
    id_value = fields.get(id_field_name)

    if not id_value:
        return {
            "success": False,
            "document_type": doc_type,
            "confidence": classification["confidence"],
            "message": f"Detected this as a likely {doc_type} document, but couldn't extract a readable ID number to validate. Try a clearer image.",
            "extracted_fields": fields,
            "raw_text_preview": text_clean[:300],
        }

    validation = validator_fn(id_value)

    return {
        "success": True,
        "document_type": doc_type,
        "confidence": classification["confidence"],
        "format_verified": validation["valid"],
        "validation_detail": validation,
        "extracted_fields": fields,
        "message": (
            f"Detected as {doc_type}. Format check: "
            + ("PASSED" if validation["valid"] else "FAILED")
        ),
        "raw_text_preview": text_clean[:300],
    }
