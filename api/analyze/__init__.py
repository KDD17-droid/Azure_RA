import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

import azure.functions as func

ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
MAX_CHARS = 5000

SUMMARY_POLL_INTERVAL_SECONDS = 2
SUMMARY_MAX_POLLS = 10


def main(req: func.HttpRequest) -> func.HttpResponse:
    if not ENDPOINT or not KEY:
        return _json_response(
            {"error": "Missing LANGUAGE_ENDPOINT / LANGUAGE_KEY"},
            500,
        )

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    text = (body.get("text") or "").strip()

    if not text:
        return _json_response({"error": "Text is required"}, 400)

    if len(text) > MAX_CHARS:
        return _json_response({"error": "Text too long"}, 400)

    try:
        lang_doc = _safe_get_first(
            _call_language("LanguageDetection", text),
            "results",
            "documents",
        )

        detected = (lang_doc.get("detectedLanguage") if lang_doc else {}) or {}
        lang_code = detected.get("iso6391Name", "en")

        sentiment = _safe_get_first(
            _call_language("SentimentAnalysis", text, lang_code),
            "results",
            "documents",
        )

        keyphrases = _safe_get_first(
            _call_language("KeyPhraseExtraction", text, lang_code),
            "results",
            "documents",
        )

        entities = _safe_get_first(
            _call_language("EntityRecognition", text, lang_code),
            "results",
            "documents",
        )

        pii = _safe_get_first(
            _call_language("PiiEntityRecognition", text, lang_code),
            "results",
            "documents",
        )

    except Exception:
        logging.exception("Azure AI Language failed")
        return _json_response({"error": "Azure AI request failed"}, 502)

    summary = None
    try:
        summary = _summarize(text, lang_code)
    except Exception:
        logging.exception("Summarization failed")

    result = {
        "sentiment": sentiment.get("sentiment") if sentiment else None,
        "confidenceScores": sentiment.get("confidenceScores") if sentiment else {},
        "keyPhrases": keyphrases.get("keyPhrases", []) if keyphrases else [],
        "entities": [
            {
                "text": e.get("text", ""),
                "category": e.get("category", "unknown"),
            }
            for e in (entities.get("entities", []) if entities else [])
        ],
        "language": {
            "name": detected.get("name"),
            "iso6391Name": lang_code,
            "confidenceScore": detected.get("confidenceScore"),
        },
        "pii": {
            "redactedText": pii.get("redactedText", text) if pii else text,
            "entities": [
                {
                    "text": e.get("text", ""),
                    "category": e.get("category", "unknown"),
                }
                for e in (pii.get("entities", []) if pii else [])
            ],
        },
        "summary": summary,
    }

    return _json_response(result, 200)


def _call_language(kind: str, text: str, language: Optional[str] = None) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"

    doc = {"id": "1", "text": text}
    if language:
        doc["language"] = language

    payload = {
        "kind": kind,
        "analysisInput": {"documents": [doc]},
        "parameters": {"modelVersion": "latest"},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode("utf-8", "ignore"))


def _summarize(text: str, language: str) -> Optional[str]:
    job_url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={API_VERSION}"

    payload = {
        "displayName": "summary",
        "analysisInput": {
            "documents": [{"id": "1", "language": language, "text": text}]
        },
        "tasks": [
            {
                "kind": "AbstractiveSummarization",
                "parameters": {"sentenceCount": 2},
            }
        ],
    }

    req = urllib.request.Request(
        job_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        op = resp.headers.get("operation-location")

    if not op:
        return None

    for _ in range(SUMMARY_MAX_POLLS):
        time.sleep(SUMMARY_POLL_INTERVAL_SECONDS)

        poll = urllib.request.Request(
            op,
            headers={"Ocp-Apim-Subscription-Key": KEY},
            method="GET",
        )

        with urllib.request.urlopen(poll, timeout=TIMEOUT_SECONDS) as r:
            job = json.loads(r.read().decode())

        if job.get("status") == "succeeded":
            docs = job["tasks"]["items"][0]["results"]["documents"]
            return docs[0]["summaries"][0]["text"]

        if job.get("status") in ("failed", "cancelled"):
            return None

    return None


def _safe_get_first(data, *keys):
    try:
        for k in keys:
            data = data.get(k, {})
        return data[0] if isinstance(data, list) else data
    except Exception:
        return {}


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )