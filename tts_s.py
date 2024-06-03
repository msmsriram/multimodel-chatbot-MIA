from gtts import gTTS

# Text to be converted to speech
text = "Hello, this is a test."

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine
speech = gTTS(text=text, lang=language, slow=False)

# Saving the converted audio in a mp3 file
speech.save("text_to_speech.mp3")
