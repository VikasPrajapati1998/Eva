import asyncio
from gtts import gTTS
from mutagen.mp3 import MP3
import os
import subprocess
import time
from deep_translator import GoogleTranslator
import speech_recognition as sr
import datetime as dt
from chatbot import ChatBot
import winsound
import threading

class Conversation:
    def __init__(self):
        self.translator = GoogleTranslator()
        self.recognizer = sr.Recognizer()
        self.chatbot = ChatBot("conversation.json")
        self.rlock = threading.RLock()

        self.start_sound = "C:\\Windows\\Media\\chimes.wav"
        self.end_sound = "C:\\Windows\\Media\\notify.wav"

        self.person_name = None

        self.hour = dt.datetime.now().hour
        if 0 <= self.hour < 12:
            self.wish = "Good Morning!"
        elif 12 <= self.hour < 18:
            self.wish = "Good Afternoon!"
        else:
            self.wish = "Good Evening!"

    async def recognize_language(self, timeout=10):
        try:
            with sr.Microphone() as source:
                message = f"""
                            Hello, {self.person_name}.
                            {self.wish}, My name is Eva.
                            Please select a language.
                            I can understand three languages: Hindi, Punjabi, and English.
                            Kindly select any one of them.
                        """
                await self.speak_text(message)
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                winsound.PlaySound(self.start_sound, winsound.SND_FILENAME)
                print("Listening for language selection...")
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognizer.recognize_google(audio, language="en-US")
                winsound.PlaySound(self.end_sound, winsound.SND_FILENAME)
                print(f"Recognized: {text}")
                return text.lower()
        except sr.UnknownValueError:
            await self.speak_text("Sorry, I could not understand what you said.")
        except sr.RequestError:
            await self.speak_text("Could not request results; please check your internet connection.")
        except Exception as e:
            print(f"Error in recognize_language: {e}")
        return None

    async def get_audio_length(self, file_path="speech.mp3"):
        try:
            audio = MP3(file_path)
            return audio.info.length - 0.2
        except Exception as e:
            print(f"Error in get_audio_length: {e}")
            return 0

    async def speak_text(self, text, lang="en"):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save("speech.mp3")
            subprocess.run(["start", "speech.mp3"], shell=True)
            await asyncio.sleep(await self.get_audio_length("speech.mp3"))
            os.remove("speech.mp3")
        except Exception as e:
            print(f"Error in speak_text: {e}")

    async def translate_text(self, text, source, target):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception as e:
            return f"Error in translation: {e}"

    async def recognize_speech(self, language="en-US", timeout=10):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                winsound.PlaySound(self.start_sound, winsound.SND_FILENAME)
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognizer.recognize_google(audio, language=language)
                print(f"Recognized: {text}")
                return text
        except sr.UnknownValueError:
            await self.speak_text("Sorry, I could not understand that.", lang="en")
        except sr.RequestError:
            await self.speak_text("Could not request results; please check your internet connection.", lang="en")
        except Exception as e:
            print(f"Error in recognize_speech: {e}")
        return None

    async def handle_conversation(self, lang_code, lang_name, exit_commands):
        await self.speak_text(f"Congratulations, you have chosen {lang_name}. How can I help you?", lang=lang_code)
        while True:
            user_input = await self.recognize_speech(language=f"{lang_code}-IN")
            if user_input and any(cmd in user_input for cmd in exit_commands):
                await self.speak_text(f"It was nice helping you. Goodbye!", lang=lang_code)
                break
            response = await self.translate_text(user_input, lang_code, "en")
            bot_response = await self.translate_text(self.chatbot.respond(response), "en", lang_code)
            await self.speak_text(bot_response, lang=lang_code)

    async def talk(self, person_name="Eva"):
        self.person_name = person_name
        recognized_language = await self.recognize_language()

        if recognized_language:
            if "hindi" in recognized_language:
                await self.handle_conversation(lang_code="hi", lang_name="Hindi", exit_commands=["बाहर निकले", "बंद करे", "धन्यवाद"])
            elif "punjabi" in recognized_language:
                await self.handle_conversation(lang_code="pa", lang_name="Punjabi", exit_commands=["ਬਾਹਰ ਜਾਓ", "ਬੰਦ ਕਰੋ", "ਧੰਨਵਾਦ"])
            elif "english" in recognized_language:
                await self.speak_text("Congratulations, you have chosen English. How can I help you?", lang="en")
            else:
                await self.speak_text("Sorry, I could not determine the language. Please try again.", lang="en")

async def main():
    conv = Conversation()
    await conv.talk("Arjun Prajapati")

if __name__ == "__main__":
    asyncio.run(main())

