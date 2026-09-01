# QUICK REFERENCE - NEW CANONICAL HIERARCHY

## File Structure

### candidates.json
Contains raw candidate-provided responses only.

```python
# READING
candidates = load_data("data/candidates.json")
answer = candidates["candidates"]["188"]["responses"]["qid_1"]["text"]["answer"]
transcript = candidates["candidates"]["188"]["responses"]["qid_1"]["speech"]["transcript"]

# WRITING
candidates["candidates"][candidate_id] = {
    "metadata": {"candidate_id": candidate_id, "created_at": None},
    "responses": {qid: {"text": {...}, "speech": {...}, "video": {...}}}
}
```

### questions.json
Contains question definitions with reference processing (lemmas, BoW, TF-IDF).

```python
# READING
questions = load_data("data/questions.json")
question_text = questions["questions"]["qid_1"]["question"]
lemmas = questions["questions"]["qid_1"]["reference"]["lemmas"]
bow = questions["questions"]["qid_1"]["reference"]["bow"]
tfidf = questions["questions"]["qid_1"]["reference"]["tfidf"]
```

### corpus.json
Contains processed candidate data with features and evaluation results.

```python
# READING QUESTION-LEVEL DATA
corpus = load_data("data/corpus.json")

# Text processing
lemmas = corpus["corpus"][candidate_id]["responses"][qid]["text"]["processed"]["lemmas"]
bow = corpus["corpus"][candidate_id]["responses"][qid]["text"]["processed"]["bow"]
tfidf = corpus["corpus"][candidate_id]["responses"][qid]["text"]["processed"]["tfidf"]

# Text features
stats = corpus["corpus"][candidate_id]["responses"][qid]["text"]["features"]["statistics"]
sentiment = corpus["corpus"][candidate_id]["responses"][qid]["text"]["features"]["sentiment"]

# Evaluation
semantic = corpus["corpus"][candidate_id]["responses"][qid]["evaluation"]["semantic"]["similarity_score"]
lexical = corpus["corpus"][candidate_id]["responses"][qid]["evaluation"]["lexical"]["keyword_coverage"]
rule_based = corpus["corpus"][candidate_id]["responses"][qid]["evaluation"]["rule_based"]["combined_score"]

# READING OVERALL DATA
overall_stats = corpus["corpus"][candidate_id]["overall"]["text"]["features"]["statistics"]
overall_sentiment = corpus["corpus"][candidate_id]["overall"]["text"]["features"]["sentiment"]
overall_semantic = corpus["corpus"][candidate_id]["overall"]["evaluation"]["semantic"]
overall_lexical = corpus["corpus"][candidate_id]["overall"]["evaluation"]["lexical"]
overall_rule_based = corpus["corpus"][candidate_id]["overall"]["evaluation"]["rule_based"]
```

## Key Function Updates

### extract_features_statistics(text)
```python
# Before: extract_features_statistics(candidates_data, candidate_id)
# After: extract_features_statistics(text)
# Returns dictionary with word_count, sentence_count, vocabulary_score, etc.

stats = extract_features_statistics("machine learning is great")
```

### extract_features_tfidf(candidate_id, corpus)
```python
# Stores results in: corpus[candidate_id]["responses"][qid]["text"]["processed"]["tfidf"]
corpus = extract_features_tfidf(candidate_id, corpus)
```

### evaluate_candidate(corpus, questions, candidate_id)
```python
# Stores evaluation in: corpus[candidate_id]["responses"][qid]["evaluation"]
# Structure:
#   - semantic.similarity_score
#   - lexical.keyword_coverage
#   - rule_based.combined_score
corpus = evaluate_candidate(corpus, questions, candidate_id)
```

### speech_to_text(audio_path, candidates_data, corpus_data, candidate_id, qid)
```python
# Stores in:
#   - candidates_data["candidates"][candidate_id]["responses"][qid]["speech"]["transcript"]
#   - corpus_data["corpus"][candidate_id]["responses"][qid]["speech"]["transcript"]
candidates_data, corpus_data = speech_to_text(audio_path, candidates_data, corpus_data, "188", "qid_1")
```

## Common Patterns

### Initialize Candidate
```python
corpus_data["corpus"][candidate_id] = {
    "metadata": {"candidate_id": candidate_id},
    "responses": {},
    "overall": {
        "text": {"features": {"statistics": {}, "sentiment": {}}},
        "speech": {"features": {}},
        "video": {"features": {}},
        "evaluation": {"semantic": {}, "lexical": {}, "rule_based": {}},
        "fusion": {"features": {}}
    }
}
```

### Initialize Response
```python
corpus_data["corpus"][candidate_id]["responses"][qid] = {
    "text": {
        "processed": {"lemmas": [], "bow": {}, "tfidf": {}},
        "features": {"statistics": {}, "sentiment": {}}
    },
    "speech": {"transcript": "", "features": {}},
    "video": {"features": {}},
    "evaluation": {
        "semantic": {},
        "lexical": {},
        "rule_based": {}
    }
}
```

### Store Processing Results
```python
qid_data = corpus_data["corpus"][candidate_id]["responses"][qid]
qid_data["text"]["processed"]["lemmas"] = lemmas_list
qid_data["text"]["processed"]["bow"] = bow_dict
qid_data["text"]["features"]["statistics"] = stats_dict
qid_data["text"]["features"]["sentiment"] = sentiment_dict
```

### Store Evaluation
```python
qid_data["evaluation"]["semantic"]["similarity_score"] = {
    "score": 0.5,
    "level": "Moderate"
}
qid_data["evaluation"]["lexical"]["keyword_coverage"] = {
    "score": 0.75,
    "matched": ["machine", "learning"],
    "missing": []
}
qid_data["evaluation"]["rule_based"]["combined_score"] = {
    "score": 0.65,
    "level": "Moderate"
}
```

## Important: Do NOT Use

❌ `corpus[candidate_id]["lemmas"]` → Use `corpus["corpus"][candidate_id]["responses"][qid]["text"]["processed"]["lemmas"]`

❌ `corpus[candidate_id]["statistics"][qid]` → Use `corpus["corpus"][candidate_id]["responses"][qid]["text"]["features"]["statistics"]`

❌ `corpus[candidate_id]["lexical_semantic_score"]` → Use `corpus["corpus"][candidate_id]["responses"][qid]["evaluation"]`

❌ `"question_wise"` nesting → Access via QID directly in responses

❌ `"combined_answer"` in candidates → Generate programmatically from responses

## Debugging Tips

### Verify Structure
```python
import json

# Check candidates structure
with open("data/candidates.json") as f:
    data = json.load(f)
    assert "candidates" in data
    assert "metadata" in data["candidates"].get(candidate_id, {})
    assert "responses" in data["candidates"].get(candidate_id, {})

# Check corpus structure
with open("data/corpus.json") as f:
    data = json.load(f)
    candidate = data["corpus"].get(candidate_id, {})
    response = candidate["responses"].get(qid, {})
    assert "text" in response
    assert "speech" in response
    assert "video" in response
    assert "evaluation" in response
```

### Print Hierarchies
```python
# Questions hierarchy
qid_data = questions["questions"]["qid_1"]
print(f"Question: {qid_data['question']}")
print(f"Lemmas: {qid_data['reference']['lemmas']}")

# Candidate response hierarchy
response = corpus["corpus"][candidate_id]["responses"][qid]
print(f"Lemmas: {response['text']['processed']['lemmas']}")
print(f"Similarity: {response['evaluation']['semantic']['similarity_score']['score']}")

# Overall hierarchy
overall = corpus["corpus"][candidate_id]["overall"]
print(f"Overall stats: {overall['text']['features']['statistics']}")
print(f"Overall evaluation: {overall['evaluation']}")
```

---

**For complete reference, see: REFACTORING_REPORT.md**
