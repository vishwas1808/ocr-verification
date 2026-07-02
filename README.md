# Document Format Checker (PAN / Aadhar / Passport)

A local Flask app that uses OCR to read uploaded PAN card, Aadhar card, or
Passport images, figures out which document type it is, and checks whether
the extracted ID number matches the **official format rules** for that
document.

## What this is — and isn't

**It does:**
- OCR the uploaded image (Tesseract)
- Classify it as PAN / Aadhar / Passport based on keywords + number patterns
- Extract fields it can read (ID number, name, DOB, gender, etc.)
- Validate the ID number's **format**:
  - PAN: `AAAAA9999A` pattern + holder-category check
  - Aadhar: 12 digits + Verhoeff checksum (the real algorithm UIDAI uses)
  - Passport: Indian passport number pattern

**It does NOT:**
- Connect to UIDAI, NSDL/Income Tax Dept, or Passport Seva — there's no
  public API for any of these, and building something that pretends to
  check against them would be misleading.
- Confirm the document is genuine, unaltered, or belongs to a real person.

So a "Format Verified" result means "structurally valid, looks like a real
document of this type" — not "confirmed authentic by the government."
This is the legitimate ceiling for what a tool like this can honestly claim,
and it's still useful for catching obviously fake/garbled/mistyped numbers.

## Setup

```bash
# 1. Install Tesseract OCR (the engine, not just the Python wrapper)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# Mac:
brew install tesseract
# Windows: download installer from
# https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Project structure

```
doc_verify/
├── app.py              # Flask backend, routes, file handling
├── document_engine.py  # OCR, classification, format validation logic
├── requirements.txt
├── templates/
│   └── index.html       # Frontend UI (HTML/CSS/JS, no framework needed)
└── uploads/             # (unused for persistent storage — files are processed in-memory)
```

## How it works

1. You upload an image (or PDF — first page only) via the browser.
2. The frontend sends it to `POST /api/verify` as multipart form data.
3. `document_engine.py`:
   - Preprocesses the image (grayscale, denoise, adaptive threshold) and
     runs Tesseract OCR.
   - Scores the extracted text against keyword sets for each document type,
     boosted by regex pattern hits (PAN/Aadhar/Passport number shapes).
   - Picks the best-matching type, extracts fields, and runs the relevant
     validator.
4. The result (document type, confidence, extracted fields, pass/fail, and
   a plain-text reason) is returned as JSON and rendered in the UI.

## Extending it

- **Add a new document type**: add keywords to `KEYWORDS`, a regex to match
  its ID format, a validator function, and wire it into `VALIDATORS` and
  `extract_fields()` in `document_engine.py`.
- **Improve OCR accuracy**: try different `cv2` preprocessing (deskewing,
  contrast stretching) for your specific image quality, or swap in a
  cloud OCR API if you later want higher accuracy than Tesseract gives you.
- **Deploy beyond your laptop**: swap Flask's dev server for `gunicorn`
  (`gunicorn -w 4 -b 0.0.0.0:5000 app:app`), and put it behind nginx with
  HTTPS if you ever expose it past localhost. Given this handles ID
  documents, don't expose it on the open internet without auth + HTTPS,
  even for a learning project.

## Limitations to know about

- OCR accuracy depends heavily on image quality — blurry, angled, or
  low-light photos will sometimes misread numbers, especially Aadhar's
  digit-only format which has no letters to anchor on.
- The Verhoeff checksum on Aadhar is mathematically real, but a random
  12-digit number has roughly a 1-in-10 chance of accidentally passing
  it — so a checksum pass is reassuring, not airtight, the same way the
  real UIDAI system treats it as just the first level of validation.
- PDF support requires PyMuPDF (`pip install PyMuPDF`); included in
  requirements.txt already.
