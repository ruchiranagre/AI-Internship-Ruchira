import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load your .wav file
with sr.AudioFile("Sample.wav") as source:
    audio = recognizer.record(source)

# Try recognizing the speech
try:
    text = recognizer.recognize_google(audio)
    print("✅ Transcription:", text)
except sr.UnknownValueError:
    print("❌ Could not understand audio.")
except sr.RequestError as e:
    print(f"❌ Could not request results; {e}")
