import speech_recognition as sr

def detect_voice():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Adjust for ambient noise with a longer duration
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=3)  # Increase duration for better noise reduction
        print("Listening for voice...")

        # Set a dynamic energy threshold for better voice detection
        recognizer.dynamic_energy_threshold = True

        try:
            # Listen for audio with a timeout of 5 seconds
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("No audio detected. Please speak louder or closer to the microphone.")
            return False

    try:
        # Convert speech to text using Google Web Speech API
        text = recognizer.recognize_google(audio)
        print(f"Detected speech: {text}")

        # Log the detected speech in malicious_activity_log.txt
        with open("malicious_activity_log.txt", "a") as log_file:
            log_file.write(f"Detected speech: {text}\n")

        return True  # Voice detected
    except sr.UnknownValueError:
        print("No speech detected. Please try again.")
        return False  # No voice detected
    except sr.RequestError:
        print("Speech recognition service failed. Please check your internet connection.")
        return False