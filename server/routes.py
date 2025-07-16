from flask import Blueprint, request, jsonify
from transformers import pipeline

api_bp = Blueprint("api", __name__)

# Load summarization pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@api_bp.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text")
    summary_type = data.get("type", "short")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Customize max/min length based on summary type
    if summary_type == "bullets":
        max_len, min_len = 90, 30
    elif summary_type == "simple":
        max_len, min_len = 70, 25
    else:
        max_len, min_len = 100, 30

    try:
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return jsonify({"summary": summary[0]["summary_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
