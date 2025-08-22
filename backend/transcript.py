import whisper

def transcribe_with_timestamps(audio_path, model_size="base"):
    """
    Transcribes audio using Whisper with word-level timestamps
    and returns sentence-level and word-level transcripts.
    """
    # Load model
    model = whisper.load_model(model_size)

    # Transcribe with word-level timestamps
    result = model.transcribe(audio_path, word_timestamps=True)

    word_level = []       # Each word with its start and end time
    sentence_level = []   # Each sentence with start, end, and text

    current_sentence = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            # Add word to word-level list
            word_level.append({
                "word": word_info["text"],
                "start": word_info["start"],
                "end": word_info["end"]
            })

            # Build sentences
            current_sentence.append(word_info)
            if word_info["text"].endswith(('.', '?', '!')):
                # End of sentence
                sentence_text = ' '.join(w['text'] for w in current_sentence)
                sentence_level.append({
                    "text": sentence_text,
                    "start": current_sentence[0]["start"],
                    "end": current_sentence[-1]["end"]
                })
                current_sentence = []

    # Handle any remaining words as last sentence
    if current_sentence:
        sentence_text = ' '.join(w['text'] for w in current_sentence)
        sentence_level.append({
            "text": sentence_text,
            "start": current_sentence[0]["start"],
            "end": current_sentence[-1]["end"]
        })

    # Save word-level transcript
    with open("transcript_word_level.txt", "w", encoding="utf-8") as f:
        for w in word_level:
            f.write(f"{w['word']} ({w['start']:.2f}-{w['end']:.2f})\n")

    # Save sentence-level transcript
    with open("transcript_sentence_level.txt", "w", encoding="utf-8") as f:
        for s in sentence_level:
            f.write(f"{s['text']} ({s['start']:.2f}-{s['end']:.2f})\n")

    return word_level, sentence_level


# Example usage
audio_file = "test.mp3"
words, sentences = transcribe_with_timestamps(audio_file)
print("Transcription completed!")
