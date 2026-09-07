import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from chatbot import create_chatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coacheye.app")

app = Flask(__name__)
CORS(app)

logger.info("Initializing chatbot...")
chatbot = create_chatbot()
logger.info("Chatbot initialized.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = chatbot({"question": user_message})
    except Exception:
        logger.exception("Chatbot query failed")
        return jsonify({"error": "Failed to process the message"}), 500

    return jsonify(
        {
            "response": response["answer"],
            "sources": response.get("sources", []),
        }
    )


if __name__ == "__main__":
    app.run(port=5000, debug=False)
