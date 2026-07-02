"""
app.py
------
Flask backend for the document OCR/format-checker tool.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import uuid
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import pytesseract

from document_engine import analyze_document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"}
MAX_FILE_SIZE_MB = 12

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_from_upload(file_storage) -> Image.Image:
    """Load an uploaded file (image or PDF first page) into a PIL Image."""
    filename = file_storage.filename
    ext = filename.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PDF support requires PyMuPDF. Install it with: pip install PyMuPDF"
            )
        pdf_bytes = file_storage.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(file_storage.stream)
        # Phone cameras often store the photo "upright" only via EXIF
        # orientation metadata, not the actual pixel data. Without this,
        # a perfectly good photo can look sideways/upside-down to OCR
        # even though it displays correctly in normal photo viewers.
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/verify", methods=["POST"])
def verify_document():
    if "document" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded. Field name must be 'document'."}), 400

    file = request.files["document"]

    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        pil_img = load_image_from_upload(file)
        result = analyze_document(pil_img)
        return jsonify(result), 200

    except pytesseract.TesseractNotFoundError:
        return jsonify({
            "success": False,
            "message": (
                "Tesseract OCR engine not found on this machine. Install it from "
                "https://github.com/UB-Mannheim/tesseract/wiki (Windows) and restart "
                "the server. If it's already installed somewhere non-standard, set "
                "the exact path in document_engine.py."
            ),
        }), 500

    except RuntimeError as e:
        return jsonify({"success": False, "message": str(e)}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Could not process this file. It may be corrupted, unreadable, or in an unsupported format.",
            "error_detail": str(e),
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "message": f"File too large. Max size is {MAX_FILE_SIZE_MB}MB."
    }), 413


if __name__ == "__main__":
    print("=" * 60)
    print(" Document Format Checker -- running locally")
    print(" Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
