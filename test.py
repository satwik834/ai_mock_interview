import whisper

audio_path = './test.mp3'

# Load Whisper model
model = whisper.load_model("base")

# Transcribe
result = model.transcribe(audio_path)

# Save transcript to file
with open("transcript.txt", "w", encoding="utf-8") as f:
    # split into sentences
    lines = result["text"].split('.')
    # write each sentence on a new line
    for line in lines:
        line = line.strip()
        if line:  # avoid empty lines
            f.write(line + ".\n")
