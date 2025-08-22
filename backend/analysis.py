import whisper_timestamped
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json

# ------------------------
# 1️⃣ Transcription
# ------------------------
def transcribe_with_words_and_sentences(audio_path, model_size="base"):
    model = whisper_timestamped.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)

    word_level = []
    sentence_level = []
    current_sentence = []

    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word_level.append({
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"]
            })
            current_sentence.append(word_info)
            if word_info["word"].endswith(('.', '?', '!')):
                sentence_text = ' '.join(w['word'] for w in current_sentence)
                sentence_level.append({
                    "text": sentence_text,
                    "start": current_sentence[0]["start"],
                    "end": current_sentence[-1]["end"]
                })
                current_sentence = []

    if current_sentence:
        sentence_text = ' '.join(w['word'] for w in current_sentence)
        sentence_level.append({
            "text": sentence_text,
            "start": current_sentence[0]["start"],
            "end": current_sentence[-1]["end"]
        })

    return word_level, sentence_level


# ------------------------
# 2️⃣ Smart Filler Detection
# ------------------------
def detect_fillers(sentence_level, filler_words):
    filler_patterns = [
        (fw, re.compile(rf"\b{re.escape(fw)}\b", re.IGNORECASE))
        for fw in filler_words
    ]
    for s in sentence_level:
        text = s['text']
        fillers_found = []
        for filler, pattern in filler_patterns:
            matches = pattern.findall(text)
            fillers_found.extend([filler] * len(matches))
        s['filler_count'] = len(fillers_found)
        s['fillers_found'] = fillers_found
    return sentence_level

def filler_summary(sentence_level):
    all_fillers = []
    total_words = 0
    for s in sentence_level:
        all_fillers.extend(s['fillers_found'])
        total_words += len(s['text'].split())
    total_fillers = len(all_fillers)
    fillers_per_100_words = round((total_fillers / total_words) * 100, 2) if total_words > 0 else 0
    most_common = Counter(all_fillers).most_common(3)
    return {
        "total_fillers": total_fillers,
        "fillers_per_100_words": fillers_per_100_words,
        "most_common_fillers": most_common
    }


# ------------------------
# 3️⃣ Disfluency Detection
# ------------------------
def analyze_disfluency(sentence_level):
    tokenizer = AutoTokenizer.from_pretrained("4i-ai/BERT_disfluency_cls")
    model = AutoModelForSequenceClassification.from_pretrained("4i-ai/BERT_disfluency_cls")
    disfluency_nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

    id2label = {0: "DISFLUENT", 1: "FLUENT"}
    for s in sentence_level:
        result = disfluency_nlp(s['text'])[0]
        pred_label_id = int(result['label'].split('_')[-1])
        s['disfluency_label'] = id2label[pred_label_id]
        s['disfluency_score'] = result['score']
    return sentence_level

def disfluency_summary(sentence_level):
    total_sentences = len(sentence_level)
    disfluent_sentences = sum(1 for s in sentence_level if s['disfluency_label'] == "DISFLUENT")
    fluent_sentences = total_sentences - disfluent_sentences
    return {
        "total_sentences": total_sentences,
        "fluent_sentences": fluent_sentences,
        "disfluent_sentences": disfluent_sentences,
        "percent_fluent": round((fluent_sentences / total_sentences) * 100, 2) if total_sentences > 0 else 0
    }


# ------------------------
# 4️⃣ Speech Metrics
# ------------------------
def compute_speech_metrics(word_level, sentence_level, pause_threshold=0.8):
    if not word_level or not sentence_level:
        return {}

    total_duration = word_level[-1]['end'] - word_level[0]['start']
    total_words = len(word_level)
    words_per_minute = round((total_words / total_duration) * 60, 2) if total_duration > 0 else 0
    total_sentences = len(sentence_level)
    avg_sentence_length = round(total_words / total_sentences, 2) if total_sentences > 0 else 0

    pauses = []
    for i in range(1, len(word_level)):
        gap = word_level[i]['start'] - word_level[i-1]['end']
        if gap >= pause_threshold:
            pauses.append(gap)
    num_pauses = len(pauses)
    avg_pause_duration = round(sum(pauses) / num_pauses, 2) if num_pauses > 0 else 0
    longest_pause = round(max(pauses), 2) if pauses else 0

    speaking_time = sum(w['end'] - w['start'] for w in word_level)
    silence_time = total_duration - speaking_time
    speaking_ratio = round((speaking_time / total_duration) * 100, 2) if total_duration > 0 else 0

    return {
        "total_duration_sec": round(total_duration, 2),
        "total_words": total_words,
        "words_per_minute": words_per_minute,
        "total_sentences": total_sentences,
        "avg_sentence_length_words": avg_sentence_length,
        "num_pauses": num_pauses,
        "avg_pause_duration_sec": avg_pause_duration,
        "longest_pause_sec": longest_pause,
        "speaking_time_sec": round(speaking_time, 2),
        "silence_time_sec": round(silence_time, 2),
        "speaking_ratio_percent": speaking_ratio
    }


# ------------------------
# 5️⃣ Overall Sentiment/Tone
# ------------------------
def analyze_sentiment_overall(sentence_level):
    """
    Safe sentiment analysis over long transcripts using truncation.
    Ensures no chunk exceeds the model's max token length.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    # Combine all sentences
    full_text = " ".join([s["text"] for s in sentence_level])

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sentiment_nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Tokenize with truncation (automatically truncates to max length)
    encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    chunk_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    # Run sentiment analysis
    result = sentiment_nlp(chunk_text)[0]

    return {
        "sentiment_label": result["label"],
        "sentiment_score": round(result["score"], 3)
    }


# ------------------------
# 6️⃣ Main Pipeline
# ------------------------
def run_interview_analysis(audio_file):
    # Step 1: Transcribe
    word_level, sentence_level = transcribe_with_words_and_sentences(audio_file)

    # Step 2: Filler detection
    filler_words_list = ["um", "uh", "like", "you know", "i mean", "so", "actually", "basically", "right"]
    sentence_level = detect_fillers(sentence_level, filler_words_list)
    filler_stats = filler_summary(sentence_level)

    # Step 3: Disfluency detection
    sentence_level = analyze_disfluency(sentence_level)
    disfluency_stats = disfluency_summary(sentence_level)

    # Step 4: Speech metrics
    speech_metrics = compute_speech_metrics(word_level, sentence_level)

    # Step 5: Overall sentiment/tone
    sentiment_stats = analyze_sentiment_overall(sentence_level)

    # Step 6: Save JSON
    output = {
        "summary": {
            "filler_stats": filler_stats,
            "disfluency_stats": disfluency_stats,
            "speech_metrics": speech_metrics,
            "sentiment_stats": sentiment_stats
        },
        "sentences": sentence_level
    }

    with open("interview_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("✅ Interview analysis complete! JSON saved as 'interview_analysis.json'")
    return output


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    audio_file = "./test.mp3"
    results = run_interview_analysis(audio_file)
    print(results["summary"])
