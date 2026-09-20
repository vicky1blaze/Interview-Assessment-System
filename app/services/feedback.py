"""
feedback.py - Explainable Rule-Based Feedback Engine.

Interprets the neural network predicted score alongside extracted multimodal
features to generate clear, actionable interview feedback detailing:
- Overall Neural Network Score & Performance Level
- Strengths
- Areas for Improvement
- Key Extracted Performance Metrics
"""

import sys

# Ensure UTF-8 console output where supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_feedback_from_metrics(overall_score: float, level: str, metrics: dict) -> str:
    """
    Generate structured, transparent feedback using rule-based interpretation
    of neural network score and multimodal metrics.
    """
    strengths = []
    improvements = []

    # 1. Content Relevance Evaluation
    cov = metrics.get("keyword_coverage", 0.0)
    sim = metrics.get("tfidf_cosine_similarity", 0.0)
    if cov >= 0.55:
        strengths.append(f"Strong concept alignment: Covered {int(cov * 100)}% of key question topics.")
    elif cov < 0.40:
        improvements.append("Concept coverage: Several expected key concepts and technical keywords were omitted.")

    if sim >= 0.50:
        strengths.append(f"High semantic focus: Solid TF-IDF alignment ({sim:.2f}) with interview themes.")
    elif sim < 0.25:
        improvements.append("Answer relevance: Responses showed low thematic overlap with reference questions.")

    # 2. Vocabulary & Articulation Evaluation
    vocab = metrics.get("vocabulary_ratio", 0.0)
    filler = metrics.get("filler_ratio", 0.0)
    avg_wc = metrics.get("word_count", 0.0)

    if vocab >= 0.65:
        strengths.append(f"Lexical diversity: High vocabulary variation ({int(vocab * 100)}% unique words).")
    elif vocab < 0.45:
        improvements.append("Word repetition: Consider varying your phrasing and vocabulary.")

    if filler <= 0.03:
        strengths.append("Articulate expression: Minimal filler words detected (<3%).")
    elif filler >= 0.07:
        improvements.append(f"Filler word reduction: High filler frequency ({filler * 100:.1f}%); practice mindful pausing.")

    if avg_wc >= 35:
        strengths.append(f"Elaboration: Substantive depth provided (average {int(avg_wc)} words per response).")
    elif avg_wc < 20 and avg_wc > 0:
        improvements.append(f"Response brevity: Answers were very short (average {int(avg_wc)} words); provide more examples.")

    # 3. Speech & Acoustic Delivery (if speech was available)
    has_speech = metrics.get("speech_available", 0.0) > 0.5
    if has_speech:
        wpm = metrics.get("speech_rate_wpm", 0.0)
        silence = metrics.get("silence_ratio", 0.0)
        rms = metrics.get("rms_energy", 0.0)

        if 120 <= wpm <= 165:
            strengths.append(f"Pacing: Optimal conversational speaking rate ({wpm:.1f} words/min).")
        elif wpm < 100 and wpm > 0:
            improvements.append(f"Pacing: Speaking rate ({wpm:.1f} WPM) was slow; aim for ~130-150 WPM.")
        elif wpm > 175:
            improvements.append(f"Pacing: Fast speech delivery ({wpm:.1f} WPM); pacing down will aid clarity.")

        if silence <= 0.20:
            strengths.append("Fluency: Continuous delivery with minimal prolonged pauses.")
        elif silence >= 0.35:
            improvements.append(f"Fluency: Noticeable hesitation periods ({silence * 100:.1f}% pause time).")

        if rms >= 0.04:
            strengths.append("Vocal clarity: Strong, audible acoustic projection.")
        elif rms < 0.02 and rms > 0:
            improvements.append("Acoustic projection: Low volume/energy; speak closer to the microphone.")
    else:
        improvements.append("Modality note: Text-only response analyzed; speech modality provides additional assessment depth.")

    # Fallbacks if list is empty
    if not strengths:
        strengths.append("Baseline communication established across all questions.")
    if not improvements:
        improvements.append("Maintain current balanced preparation and confident delivery.")

    # Format into professional human-readable report
    border = "=" * 65
    divider = "-" * 65
    lines = [
        "",
        border,
        "             COMPREHENSIVE INTERVIEW ASSESSMENT REPORT",
        border,
        f"  Overall Score   : {overall_score:.2f} / 100",
        f"  Performance Band: {level}",
        divider,
        "  KEY STRENGTHS:",
    ]
    for s in strengths:
        lines.append(f"    [+] {s}")

    lines.append("")
    lines.append("  AREAS FOR IMPROVEMENT:")
    for imp in improvements:
        lines.append(f"    [-] {imp}")

    lines.append(divider)
    lines.append("  EXTRACTED PERFORMANCE METRICS:")
    lines.append(f"  - Average Word Count       : {avg_wc:.1f}")
    lines.append(f"  - Vocabulary Diversity     : {vocab * 100:.1f}%")
    lines.append(f"  - Filler Word Ratio        : {filler * 100:.1f}%")
    lines.append(f"  - Key Concept Coverage     : {cov * 100:.1f}%")
    lines.append(f"  - TF-IDF Cosine Similarity : {sim:.4f}")
    if has_speech:
        lines.append(f"  - Speech Speaking Rate     : {metrics.get('speech_rate_wpm', 0.0):.1f} WPM")
        lines.append(f"  - Silence & Pause Ratio    : {metrics.get('silence_ratio', 0.0) * 100:.1f}%")
        lines.append(f"  - Acoustic RMS Energy      : {metrics.get('rms_energy', 0.0):.4f}")
    lines.append(border)
    lines.append("")

    report = "\n".join(lines)
    return report

def feedback(candidate_id: str, corpus_data: dict) -> str:
    """
    Display comprehensive feedback for a candidate from the canonical corpus hierarchy.
    """
    candidate = corpus_data.get("corpus", {}).get(candidate_id, {})
    if not candidate:
        msg = f"Candidate '{candidate_id}' not found in corpus."
        print(msg)
        return msg

    overall = candidate.get("overall", {})
    nn_eval = overall.get("evaluation", {}).get("neural_network", {})
    score = nn_eval.get("score", 0.0)
    level = nn_eval.get("level", "Unrated")

    # Aggregate metrics from overall or responses
    stats = overall.get("text", {}).get("features", {}).get("statistics", {})
    speech_feat = overall.get("speech", {}).get("features", {})
    lexical = overall.get("evaluation", {}).get("lexical", {}).get("keyword_coverage", {})
    semantic = overall.get("evaluation", {}).get("semantic", {}).get("similarity_score", {})
    sentiment = overall.get("text", {}).get("features", {}).get("sentiment", {})

    metrics = {
        "word_count": float(stats.get("word_count", 0)),
        "vocabulary_ratio": float(stats.get("vocabulary_ratio", 0.0)),
        "filler_ratio": float(stats.get("filler_ratio", 0.0)),
        "keyword_coverage": float(lexical.get("score", 0.0)),
        "tfidf_cosine_similarity": float(semantic.get("score", 0.0)),
        "sentiment_pos": float(sentiment.get("pos", 0.0)),
        "sentiment_compound": float(sentiment.get("compound", 0.0)),
        "speech_rate_wpm": float(speech_feat.get("speech_rate_wpm", 0.0)),
        "silence_ratio": float(speech_feat.get("silence_ratio", 0.0)),
        "rms_energy": float(speech_feat.get("rms_energy", 0.0)),
        "speech_available": float(speech_feat.get("speech_available", 0.0))
    }

    report = generate_feedback_from_metrics(score, level, metrics)
    print(report)
    return report
