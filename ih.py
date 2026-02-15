import customtkinter as ctk
import cv2
import threading
import time
import webbrowser
import numpy as np
import speech_recognition as sr
import pyttsx3  
from PIL import Image
from tkinter import messagebox
from transformers import pipeline
from deep_translator import GoogleTranslator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class NeuralCore:
    def __init__(self, status_callback):
        self.model = None
        self.status_callback = status_callback
        self.engine = pyttsx3.init()
        threading.Thread(target=self._load_model, daemon=True).start()

    def speak(self, text):
        def _say():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=_say, daemon=True).start()

    def _load_model(self):
        try:
            time.sleep(1)
            self.status_callback("Loading...")
            self.model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
            self.status_callback("AI SECURE ONLINE")
            self.speak("Scanning your face")
        except Exception as e:
            self.status_callback("AI ERROR!")

    def analyze_text(self, text):
        hard_blacklist = ["плохоеслово", "роблокс", "убью", "кровь"]
        if any(word in text.lower() for word in hard_blacklist):
            return False, "REASON: Blacklist (Manual Ban)"
        if not self.model:
            return False, "AI is loading..."

        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            translated_text = text

        try:
            results = self.model(translated_text)
        except Exception as e:
            return False, f"Model error: {e}"

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]
        if isinstance(results, dict):
            results = [results]

        bad_labels = ['toxic', 'severe_toxic', 'threat', 'insult', 'obscene']
        found_threats = []
        for r in results:
            label = r.get('label') if isinstance(r, dict) else None
            score = r.get('score', 0) if isinstance(r, dict) else 0
            if label in bad_labels and score > 0.2:
                found_threats.append(f"{label} ({int(score*100)}%)")

        if found_threats:
            return False, " | ".join(found_threats)

        return True, "Safe"

class TazaNet (ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TazaNet AI")
        self.geometry("1400x900")
        
        self.user_type = "SCANNING"
        self.is_listening = False
        
        self.setup_ui()
        self.ai = NeuralCore(self.update_ai_status)
        
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = sr.Recognizer()
        
        threading.Thread(target=self.vision_engine, daemon=True).start()

    def setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#080808")
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="TazaNet AI", font=("Impact", 35), text_color="#FF0000").pack(pady=40)
        self.cam_label = ctk.CTkLabel(self.sidebar, text="", width=260, height=200, fg_color="#111")
        self.cam_label.pack(pady=10)
        
        self.status_lbl = ctk.CTkLabel(self.sidebar, text="BOOTING...", font=("Consolas", 12))
        self.status_lbl.pack(pady=5)
        
        self.mode_lbl = ctk.CTkLabel(self.sidebar, text="DETECTING...", font=("Arial", 18, "bold"))
        self.mode_lbl.pack(pady=20)

        self.main = ctk.CTkFrame(self, fg_color="#0F0F0F")
        self.main.pack(side="right", expand=True, fill="both", padx=20, pady=20)
        
        ctk.CTkLabel(self.main, text="STAY PROTECTED", font=("Arial", 30, "bold")).pack(pady=40)
        
        self.search_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.search_frame.pack(pady=20)

        self.search_entry = ctk.CTkEntry(self.search_frame, width=600, height=60, placeholder_text="Type or use Voice...", font=("Arial", 18))
        self.search_entry.pack(side="left", padx=10)

        self.voice_btn = ctk.CTkButton(self.search_frame, text="🎤", width=60, height=60, font=("Arial", 25), 
                                       fg_color="#333", hover_color="#FF0000", command=self.start_voice)
        self.voice_btn.pack(side="left")
        
        ctk.CTkButton(self.main, text="WATCH SECURELY", width=200, height=50, fg_color="#FF0000", font=("Arial", 14, "bold"), command=self.process).pack(pady=10)
        
        self.log_box = ctk.CTkTextbox(self.main, width=800, height=400, font=("Consolas", 13))
        self.log_box.pack(pady=20)

    def update_ai_status(self, text):
        self.status_lbl.configure(text=text)

    def start_voice(self):
        if not self.is_listening:
            threading.Thread(target=self.listen, daemon=True).start()

    def listen(self):
        self.is_listening = True
        self.voice_btn.configure(fg_color="#FF0000", text="...")
        
        with sr.Microphone() as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                query = self.recognizer.recognize_google(audio, language="ru-RU")
                self.search_entry.delete(0, "end")
                self.search_entry.insert(0, query)
                self.process()
            except:
                self.ai.speak("Could not understand audio.")
            finally:
                self.is_listening = False
                self.voice_btn.configure(fg_color="#333", text="🎤")

    def vision_engine(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 6)
            
            self.user_type = "ADULT"
            for (x, y, w, h) in faces:
                face_ratio = h / w
                if face_ratio < 1.15 and w < 180:
                    self.user_type = "CHILD"
                color = (0, 255, 0) if self.user_type == "ADULT" else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            try:
                self.mode_lbl.configure(text=f"STATUS: {self.user_type}", text_color="#2ECC71" if self.user_type=="ADULT" else "#3B8ED0")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(260, 200))
                self.cam_label.configure(image=ctk_img)
            except:
                break
            time.sleep(0.03)

    def process(self):
        query = self.search_entry.get()
        if not query: return
        
        self.log_box.insert("end", f"\n[SCAN]: '{query}'\n")
        
        is_safe, reason = self.ai.analyze_text(query)
        
        if not is_safe:
            is_critical = "Blacklist" in reason or "severe_toxic" in reason
            
            if self.user_type == "CHILD" or is_critical:
                self.log_box.insert("end", f"!!! ACCESS DENIED: {reason}\n", "red")
                self.ai.speak("Access denied. Content is dangerous.")
                messagebox.showwarning("TazaNet", f"BLOCKED!\nReason: {reason}")
                return
            else:
                self.log_box.insert("end", f">>> WARNING: {reason} (Allowed for Adult)\n", "orange")
        
        self.log_box.insert("end", ">>> VERDICT: CLEAN. Access Granted.\n")
        self.ai.speak("Access granted.")
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")

if __name__ == "__main__":
    app = TazaNet()
    app.mainloop()
