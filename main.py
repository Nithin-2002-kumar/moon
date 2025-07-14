import os
import sys
import time
import json
import threading
import subprocess
import webbrowser
import logging
import tkinter as tk
from tkinter import ttk, Canvas, messagebox, scrolledtext
from PIL import Image, ImageTk
import speech_recognition as sr
import pyttsx3
import pyautogui
import psutil
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import wikipedia
import requests
import wolframalpha
import spacy
import pygame
import smtplib
from email.mime.text import MIMEText
import face_recognition
from textblob import TextBlob
from deepface import DeepFace
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(filename="moon.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
CONFIG_FILE = "moon_config.json"
DEFAULT_CONFIG = {
    "wake_word": "moon",
    "hotword": "moon",
    "speech_rate": 170,
    "voice": 0,  # Male voice
    "weather_api_key": "YOUR_OPENWEATHERMAP_API_KEY",
    "wolfram_api_key": "YOUR_WOLFRAM_ALPHA_KEY",
    "openai_api_key": "YOUR_OPENAI_API_KEY",
    "email_user": "your_email@gmail.com",
    "email_pass": "your_app_password",
    "known_faces": {},
    "use_memory": True,
    "default_user": {"speech_rate": 170, "greeting": "sir"}
}

# Define categories for classification
LIVING_ORGANISMS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "fish"]
HUMANS = ["person"]
NON_LIVING = ["bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "chair", "couch", "table", "bed", "tv",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Color definitions in HSV
COLOR_RANGES = {
    "red": [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
    "orange": [(10, 50, 50), (25, 255, 255)],
    "yellow": [(25, 50, 50), (35, 255, 255)],
    "green": [(35, 50, 50), (85, 255, 255)],
    "blue": [(85, 50, 50), (125, 255, 255)],
    "purple": [(125, 50, 50), (150, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "black": [(0, 0, 0), (180, 255, 30)],
    "gray": [(0, 0, 50), (180, 30, 200)]
}

class MOON:
    def __init__(self):
        self.load_config()
        self.root = tk.Tk()
        self.root.title("MOON Interface with Grok")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.9)  # Slightly more opaque
        self.root.configure(bg='#1a1a1a')  # Darker gray background
        self.running = True
        self.listening = False
        self.wake_detected = False
        self.processes = {}
        self.current_user = None

        # Initialize components
        self.init_components()
        self.init_ui()
        self.start_threads()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG
            self.save_config()

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def init_components(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[self.config["voice"]].id)
        self.engine.setProperty('rate', self.config["speech_rate"])

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', probability=True)
        self.train_classifier()

        self.memory = {"history": {}, "tasks": [], "reminders": [], "learned_commands": {}}
        self.load_memory()

        pygame.init()

    def load_memory(self):
        if os.path.exists("moon_memory.json"):
            with open("moon_memory.json", 'r') as f:
                self.memory = json.load(f)
        if not isinstance(self.memory["history"], dict):
            self.memory["history"] = {}

    def save_memory(self):
        if self.config["use_memory"]:
            with open("moon_memory.json", 'w') as f:
                json.dump(self.memory, f, indent=4)

    def init_ui(self):
        # Main canvas
        self.canvas = Canvas(self.root, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Title bar
        self.title_label = tk.Label(self.canvas, text="MOON", font=('Orbitron', 20, 'bold'),
                                    fg='#00ffff', bg='#1a1a1a')
        self.canvas.create_window(self.root.winfo_screenwidth() // 2, 30, window=self.title_label)

        # Status panel
        self.status_frame = tk.Frame(self.canvas, bg='#2a2a2a', bd=2, relief='ridge')
        self.status_var = tk.StringVar(value=f"Status: Waiting for '{self.config['wake_word']}'")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_var, fg='#00ffff', bg='#2a2a2a',
                                     font=('Orbitron', 12))
        self.status_label.pack(pady=5)
        self.user_var = tk.StringVar(value="User: Unknown")
        self.user_label = tk.Label(self.status_frame, textvariable=self.user_var, fg='#00ffff', bg='#2a2a2a',
                                   font=('Orbitron', 12))
        self.user_label.pack(pady=5)
        self.sys_var = tk.StringVar(value="System: CPU 0% | Mem 0%")
        self.sys_label = tk.Label(self.status_frame, textvariable=self.sys_var, fg='#00ffff', bg='#2a2a2a',
                                  font=('Orbitron', 12))
        self.sys_label.pack(pady=5)
        self.canvas.create_window(150, 150, window=self.status_frame, anchor='nw')

        # Command log
        self.log_frame = tk.Frame(self.canvas, bg='#2a2a2a', bd=2, relief='ridge')
        self.command_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=80, bg='#1a1a1a', fg='#00ffff',
                                                      font=('Orbitron', 12), bd=0, wrap=tk.WORD)
        self.command_text.pack(padx=10, pady=10)
        self.canvas.create_window(self.root.winfo_screenwidth() // 2, self.root.winfo_screenheight() // 2,
                                  window=self.log_frame)

        # Control panel
        self.control_frame = tk.Frame(self.canvas, bg='#2a2a2a', bd=2, relief='ridge')
        style = ttk.Style()
        style.configure('MOON.TButton', font=('Orbitron', 10), foreground='#00ffff', background='#1a1a1a',
                        borderwidth=1, relief='flat')
        style.map('MOON.TButton', background=[('active', '#00cccc')])

        self.listen_btn = ttk.Button(self.control_frame, text="🎤 Listen", command=self.toggle_listening,
                                     style='MOON.TButton')
        self.listen_btn.grid(row=0, column=0, padx=5, pady=5)
        self.color_btn = ttk.Button(self.control_frame, text="🌈 Detect Colors", command=lambda: self.process_command("detect colors"),
                                    style='MOON.TButton')
        self.color_btn.grid(row=0, column=1, padx=5, pady=5)
        self.classify_btn = ttk.Button(self.control_frame, text="🔍 Classify Entities",
                                       command=lambda: self.process_command("classify entities"), style='MOON.TButton')
        self.classify_btn.grid(row=0, column=2, padx=5, pady=5)
        self.exit_btn = ttk.Button(self.control_frame, text="⏹ Shutdown", command=self.shutdown, style='MOON.TButton')
        self.exit_btn.grid(row=0, column=3, padx=5, pady=5)
        self.canvas.create_window(self.root.winfo_screenwidth() - 200, 80, window=self.control_frame, anchor='ne')

        # Animated HUD
        self.hud_elements = []
        for i in range(12):
            arc = self.canvas.create_arc(50 + i * 40, 50 + i * 40, 150 + i * 40, 150 + i * 40,
                                         start=np.random.randint(0, 360), extent=90, outline='#00ffff', style='arc', width=2)
            self.hud_elements.append(arc)
        self.animate_hud()

        # Detection results panel
        self.results_frame = tk.Frame(self.canvas, bg='#2a2a2a', bd=2, relief='ridge')
        self.results_var = tk.StringVar(value="Detection Results: None")
        self.results_label = tk.Label(self.results_frame, textvariable=self.results_var, fg='#00ffff', bg='#2a2a2a',
                                      font=('Orbitron', 12), wraplength=300, justify='left')
        self.results_label.pack(padx=10, pady=10)
        self.canvas.create_window(self.root.winfo_screenwidth() - 150, self.root.winfo_screenheight() - 150,
                                  window=self.results_frame, anchor='se')

    def animate_hud(self):
        if self.running:
            for arc in self.hud_elements:
                coords = self.canvas.coords(arc)
                self.canvas.move(arc, np.random.randint(-1, 2), np.random.randint(-1, 2))
                angle = np.random.randint(0, 360)
                self.canvas.itemconfig(arc, start=angle)
                if coords[2] > self.root.winfo_screenwidth() or coords[3] > self.root.winfo_screenheight():
                    self.canvas.coords(arc, 50, 50, 150, 150)
            self.root.after(100, self.animate_hud)

    def train_classifier(self):
        commands = [
            "open notepad", "open browser", "search web", "set reminder", "tell time", "tell date",
            "take screenshot", "send email", "check weather", "scan room", "detect objects",
            "recognize face", "analyze sentiment", "detect emotion", "close notepad", "type in notepad",
            "navigate browser", "detect gender", "ask grok", "who am i", "classify entities", "detect colors"
        ]
        labels = ["open", "open", "search", "reminder", "time", "date", "screenshot", "email", "weather", "scan",
                  "object_detection", "face_recognition", "sentiment", "emotion", "close", "type", "navigate", "gender",
                  "grok", "identify", "classify", "color_detection"]
        X = self.vectorizer.fit_transform(commands)
        self.classifier.fit(X, labels)

    def speak(self, text):
        user_prefs = self.get_user_prefs()
        self.engine.setProperty('rate', user_prefs["speech_rate"])
        self.command_text.insert(tk.END, f"MOON: {text}\n")
        self.command_text.see(tk.END)
        if self.config["use_memory"] and self.current_user:
            if self.current_user not in self.memory["history"]:
                self.memory["history"][self.current_user] = []
            self.memory["history"][self.current_user].append({"role": "MOON", "text": text, "time": datetime.now().isoformat()})
            self.save_memory()
        self.engine.say(text)
        self.engine.runAndWait()

    def get_user_prefs(self):
        if self.current_user and self.current_user in self.config["known_faces"]:
            return self.config["known_faces"][self.current_user]["prefs"]
        return self.config["default_user"]

    def identify_user(self):
        cap = cv2.VideoCapture(0)
        known_encodings = [data["encoding"] for data in self.config["known_faces"].values()]
        known_names = list(self.config["known_faces"].keys())
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "Unknown"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                cap.release()
                self.user_var.set(f"User: {known_names[first_match_index]}")
                return known_names[first_match_index]
        cap.release()
        self.user_var.set("User: Unknown")
        return "Unknown"

    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.status_var.set("Status: Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio).lower()
                if self.config["hotword"] in text:
                    self.current_user = self.identify_user()
                    user_prefs = self.get_user_prefs()
                    greeting = user_prefs["greeting"]
                    self.speak(f"At your service, {greeting}. Powered by Grok.")
                    command = text.replace(self.config["hotword"], "").strip()
                    self.command_text.insert(tk.END, f"{greeting.capitalize()}: {command}\n")
                    if self.config["use_memory"] and self.current_user:
                        if self.current_user not in self.memory["history"]:
                            self.memory["history"][self.current_user] = []
                        self.memory["history"][self.current_user].append({"role": "user", "text": command, "time": datetime.now().isoformat()})
                    return command
            except sr.UnknownValueError:
                self.speak("I didn't catch that.")
            except sr.RequestError:
                self.speak("Speech service is unavailable.")
            except sr.WaitTimeoutError:
                pass
            return None

    def wake_word_listener(self):
        def callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio).lower()
                if self.config["wake_word"] in text:
                    self.wake_detected = True
                    user_prefs = self.get_user_prefs()
                    self.speak(f"Wake word '{self.config['wake_word']}' detected. Listening for your command, {user_prefs['greeting']}...")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                self.speak("Speech recognition service is down.")

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.status_var.set(f"Status: Waiting for '{self.config['wake_word']}'")
            listener = self.recognizer.listen_in_background(source, callback)
            while self.running:
                if self.wake_detected:
                    listener.stop()
                    command = self.listen()
                    if command:
                        self.process_command(command)
                    self.wake_detected = False
                    listener = self.recognizer.listen_in_background(source, callback)
                time.sleep(0.1)

    def toggle_listening(self):
        if not self.listening:
            self.listening = True
            self.listen_btn.config(text="🔴 Stop")
            self.status_var.set("Status: Manual Listening...")
            threading.Thread(target=self.listen_loop, daemon=True).start()
        else:
            self.listening = False
            self.listen_btn.config(text="🎤 Listen")
            self.status_var.set(f"Status: Waiting for '{self.config['wake_word']}'")

    def process_command(self, command):
        X = self.vectorizer.transform([command])
        intent = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])

        if confidence < 0.6:
            self.handle_unknown_command(command)
            return

        user_prefs = self.get_user_prefs()
        greeting = user_prefs["greeting"]

        if intent == "open":
            self.open_app(command)
        elif intent == "close":
            self.close_app(command)
        elif intent == "type":
            self.type_in_app(command)
        elif intent == "navigate":
            self.navigate_browser(command)
        elif intent == "search":
            self.search_web(command)
        elif intent == "reminder":
            self.set_reminder(command)
        elif intent == "time":
            self.tell_time()
        elif intent == "date":
            self.tell_date()
        elif intent == "screenshot":
            self.take_screenshot()
        elif intent == "email":
            self.send_email(command)
        elif intent == "weather":
            self.check_weather(command)
        elif intent == "scan":
            self.scan_environment()
        elif intent == "object_detection":
            self.detect_objects()
        elif intent == "face_recognition":
            self.recognize_faces()
        elif intent == "sentiment":
            self.analyze_sentiment(command)
        elif intent == "emotion":
            self.detect_emotion()
        elif intent == "gender":
            self.detect_gender()
        elif intent == "identify":
            self.speak(f"You are {self.current_user if self.current_user else 'Unknown'}, {greeting}.")
        elif intent == "classify":
            self.classify_entities()
        elif intent == "color_detection":
            self.detect_colors()
        elif intent == "grok" or intent not in ["open", "close", "type", "navigate", "search", "reminder", "time", "date",
                                                "screenshot", "email", "weather", "scan", "object_detection",
                                                "face_recognition", "sentiment", "emotion", "gender", "identify",
                                                "classify", "color_detection"]:
            self.speak("Passing to Grok for advanced processing...")
            response = self.get_grok_response(command)
            self.speak(response)
        elif "gesture" in command:
            self.gesture_control()

    @staticmethod
    def preprocess_frame(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)
        return frame

    def detect_colors(self):
        self.speak("Detecting colors, optimized for nighttime.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        roi = hsv_frame[y:y+h, x:x+w]
                        if roi.size == 0:
                            continue

                        color = self.get_dominant_color(roi)
                        class_name = self.classes[class_id]
                        label = f"{class_name}: {color} ({confidence:.2f})"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        result = f"{class_name} in {color} at ({x}, {y})"
                        results.append(result)
                        self.speak(f"Detected a {class_name} in {color} at {x}, {y}, {self.get_user_prefs()['greeting']}.")
                        self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))  # Show last 5 results

            cv2.imshow("Color Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_dominant_color(hsv_roi):
        mask = cv2.inRange(hsv_roi, (0, 30, 30), (180, 255, 255))
        hsv_pixels = hsv_roi[mask > 0]
        if len(hsv_pixels) == 0:
            return "unknown"
        hue_avg = np.mean(hsv_pixels[:, 0])
        for color_name, ranges in COLOR_RANGES.items():
            if len(ranges) == 4:
                if (ranges[0] <= hue_avg <= ranges[1]) or (ranges[2] <= hue_avg <= ranges[3]):
                    return color_name
            else:
                if ranges[0] <= hue_avg <= ranges[1]:
                    return color_name
        return "unknown"

    def classify_entities(self):
        self.speak("Classifying entities with color detection.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi = hsv_frame[y:y+h, x:x+w]
                color = self.get_dominant_color(roi)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Human: {color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                result = f"Human in {color} at ({x}, {y})"
                results.append(result)
                self.speak(f"Detected a human in {color} at {x}, {y}, {self.get_user_prefs()['greeting']}.")
                self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        class_name = self.classes[class_id]
                        roi = hsv_frame[y:y+h, x:x+w]
                        color = self.get_dominant_color(roi)
                        category = "Non-living thing"
                        color_box = (0, 0, 255)
                        if class_name in HUMANS:
                            category = "Human"
                            color_box = (0, 255, 0)
                        elif class_name in LIVING_ORGANISMS:
                            category = "Living organism"
                            color_box = (255, 0, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                        label = f"{category}: {class_name} ({color}, {confidence:.2f})"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                        result = f"{category}: {class_name} in {color} at ({x}, {y})"
                        results.append(result)
                        self.speak(f"Detected a {category.lower()} ({class_name}) in {color} at {x}, {y}, {self.get_user_prefs()['greeting']}.")
                        self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))

            cv2.imshow("Entity Classification with Colors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def open_app(self, command):
        apps = {"notepad": "notepad.exe", "browser": "https://www.google.com", "calculator": "calc.exe"}
        for app, path in apps.items():
            if app in command:
                if app == "browser":
                    webbrowser.open(path)
                    self.processes[app] = None
                else:
                    process = subprocess.Popen(path)
                    self.processes[app] = process
                self.speak(f"Opening {app}, {self.get_user_prefs()['greeting']}.")
                return
        self.speak("Application not recognized.")

    def close_app(self, command):
        apps = {"notepad": "notepad.exe", "browser": "browser", "calculator": "calc.exe"}
        for app in apps:
            if app in command:
                if app in self.processes and self.processes[app]:
                    self.processes[app].terminate()
                    del self.processes[app]
                    self.speak(f"Closing {app}, {self.get_user_prefs()['greeting']}.")
                elif app == "browser":
                    pyautogui.hotkey('alt', 'f4')
                    self.speak(f"Closing {app}, {self.get_user_prefs()['greeting']}.")
                else:
                    self.speak(f"{app} is not open, {self.get_user_prefs()['greeting']}.")
                return
        self.speak("Application not recognized.")

    def type_in_app(self, command):
        if "notepad" in command:
            if "notepad" not in self.processes:
                self.open_app("open notepad")
                time.sleep(1)
            text = command.replace("type in notepad", "").strip()
            pyautogui.write(text)
            self.speak(f"Typed '{text}' in Notepad, {self.get_user_prefs()['greeting']}.")
        else:
            self.speak("I can only type in Notepad for now.")

    def navigate_browser(self, command):
        if "browser" not in self.processes:
            self.open_app("open browser")
            time.sleep(2)
        if "back" in command:
            pyautogui.hotkey('alt', 'left')
            self.speak("Going back in browser.")
        elif "forward" in command:
            pyautogui.hotkey('alt', 'right')
            self.speak("Going forward in browser.")
        elif "refresh" in command:
            pyautogui.hotkey('f5')
            self.speak("Refreshing browser.")
        else:
            self.speak("Navigation command not recognized.")

    def search_web(self, command):
        query = command.replace("search", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        self.processes["browser"] = None
        self.speak(f"Searching for {query}, {self.get_user_prefs()['greeting']}.")

    def set_reminder(self, command):
        doc = self.nlp(command)
        date_ent = next((ent.text for ent in doc.ents if ent.label_ == "DATE"), None)
        time_ent = next((ent.text for ent in doc.ents if ent.label_ == "TIME"), "in 1 hour")
        if date_ent:
            try:
                reminder_date = datetime.strptime(date_ent, "%Y-%m-%d")
            except:
                reminder_date = datetime.now()
        else:
            reminder_date = datetime.now()
        if "in" in time_ent and "hour" in time_ent:
            hours = int(time_ent.split()[1])
            reminder_time = reminder_date + timedelta(hours=hours)
        else:
            reminder_time = reminder_date + timedelta(hours=1)
        message = command.replace(time_ent, "").replace("set reminder", "").replace(date_ent or "", "").strip()
        self.memory["reminders"].append({"text": message, "time": reminder_time.isoformat(), "user": self.current_user or "Unknown"})
        self.save_memory()
        threading.Thread(target=self.check_reminder, args=(message, reminder_time, self.current_user or "Unknown"), daemon=True).start()
        self.speak(f"Reminder set for {time_ent} on {reminder_date.strftime('%Y-%m-%d')}: {message}, {self.get_user_prefs()['greeting']}.")

    def check_reminder(self, message, reminder_time, user):
        while self.running and datetime.now() < reminder_time:
            time.sleep(10)
        if self.running and (self.current_user == user or user == "Unknown"):
            self.speak(f"Reminder, {self.get_user_prefs()['greeting']}: {message}")

    def tell_time(self):
        time_str = datetime.now().strftime("%I:%M %p")
        self.speak(f"The time is {time_str}, {self.get_user_prefs()['greeting']}.")

    def tell_date(self):
        date_str = datetime.now().strftime("%B %d, %Y")
        self.speak(f"Today is {date_str}, {self.get_user_prefs()['greeting']}.")

    def take_screenshot(self):
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        pyautogui.screenshot(filename)
        self.speak(f"Screenshot saved as {filename}, {self.get_user_prefs()['greeting']}.")

    def send_email(self, command):
        doc = self.nlp(command)
        to_person = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None)
        if not to_person:
            self.speak("Who should I send the email to?")
            return
        message = command.replace("send email to", "").replace(to_person, "").strip()
        try:
            msg = MIMEText(message)
            msg['Subject'] = "Message from MOON"
            msg['From'] = self.config["email_user"]
            msg['To'] = f"{to_person}@example.com"
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.config["email_user"], self.config["email_pass"])
            server.send_message(msg)
            server.quit()
            self.speak(f"Email sent to {to_person}, {self.get_user_prefs()['greeting']}.")
        except Exception as e:
            self.speak(f"Failed to send email: {str(e)}")

    def check_weather(self, command):
        if not self.config["weather_api_key"]:
            self.speak("Weather API key not configured.")
            return
        doc = self.nlp(command)
        location = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), "current location")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.config['weather_api_key']}&units=metric"
        try:
            response = requests.get(url).json()
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            self.speak(f"The weather in {location} is {desc} with a temperature of {temp}°C, {self.get_user_prefs()['greeting']}.")
        except:
            self.speak("Unable to fetch weather data.")

    def gesture_control(self):
        self.speak("Entering gesture control mode.")
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    thumb_tip = hand_landmarks.landmark[4].y
                    index_tip = hand_landmarks.landmark[8].y
                    if index_tip < hand_landmarks.landmark[6].y:
                        pyautogui.scroll(10)
                    elif index_tip > hand_landmarks.landmark[6].y:
                        pyautogui.scroll(-10)
                    elif abs(thumb_tip - index_tip) < 0.05:
                        pyautogui.click()
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.speak("Gesture control deactivated.")

    def scan_environment(self):
        self.speak("Scanning environment.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                result = f"Human at ({x}, {y})"
                results.append(result)
                self.speak(f"Detected a human at coordinates {x}, {y}, {self.get_user_prefs()['greeting']}.")
                self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))
            cv2.imshow("Environment Scan", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_objects(self):
        self.speak("Detecting objects.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{self.classes[class_id]}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        result = f"{self.classes[class_id]} at ({x}, {y})"
                        results.append(result)
                        self.speak(f"Detected {self.classes[class_id]} at {x}, {y}, {self.get_user_prefs()['greeting']}.")
                        self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def recognize_faces(self):
        self.speak("Recognizing faces.")
        cap = cv2.VideoCapture(0)
        known_encodings = [data["encoding"] for data in self.config["known_faces"].values()]
        known_names = list(self.config["known_faces"].keys())
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    self.current_user = name
                    self.user_var.set(f"User: {name}")
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                result = f"Face: {name} at ({left}, {top})"
                results.append(result)
                self.speak(f"Recognized {name}, {self.get_user_prefs()['greeting']}.")
                self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def analyze_sentiment(self, command):
        self.speak("Analyzing sentiment.")
        blob = TextBlob(command)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            self.speak(f"The sentiment is positive with a score of {sentiment:.2f}, {self.get_user_prefs()['greeting']}.")
        elif sentiment < 0:
            self.speak(f"The sentiment is negative with a score of {sentiment:.2f}, {self.get_user_prefs()['greeting']}.")
        else:
            self.speak("The sentiment is neutral.")

    def detect_emotion(self):
        self.speak("Detecting emotions.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                results.append(f"Emotion: {emotion}")
                self.speak(f"Detected emotion: {emotion}, {self.get_user_prefs()['greeting']}.")
                self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))
            except:
                self.speak("Unable to detect emotion.")
            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_gender(self):
        self.speak("Detecting gender.")
        cap = cv2.VideoCapture(0)
        results = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                result = DeepFace.analyze(frame, actions=['gender'], enforce_detection=False)
                gender = result[0]['dominant_gender']
                cv2.putText(frame, gender, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                results.append(f"Gender: {gender}")
                self.speak(f"Detected gender: {gender}, {self.get_user_prefs()['greeting']}.")
                self.results_var.set(f"Detection Results:\n" + "\n".join(results[-5:]))
            except:
                self.speak("Unable to detect gender.")
            cv2.imshow("Gender Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def get_grok_response(self, command):
        current_date = "April 10, 2025"
        user_history = self.memory["history"].get(self.current_user, []) if self.current_user else []
        if self.config["use_memory"] and user_history:
            recent_history = " ".join([entry["text"] for entry in user_history[-5:] if entry["role"] == "user"])
            context = f"Previous conversation: {recent_history}. Current command: {command}"
        else:
            context = command

        if "who deserves the death penalty" in command.lower() or "who deserves to die" in command.lower():
            return "As an AI, I’m not allowed to make that choice. It’s a human decision beyond my scope."
        elif "what happened yesterday" in command.lower():
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%B %d, %Y")
            return f"Yesterday was {yesterday}. I’d need more details to tell you what happened—care to specify an event or topic?"
        elif "search the web" in command.lower():
            query = command.replace("search the web", "").strip()
            return f"Simulating a web search for '{query}'. Imagine I found: '{query} is trending today on April 10, 2025!' What else can I do for you?"
        elif "analyze x post" in command.lower():
            return "I’d love to analyze an X post for you. Please provide the content or a link, and I’ll give you my take—without judging misinformation, of course."
        elif "forget our chat" in command.lower():
            if self.config["use_memory"] and self.current_user:
                self.memory["history"][self.current_user] = []
                self.save_memory()
                return "I’ve forgotten our prior chats. We’re starting fresh—how can I assist now?"
            else:
                return "Memory is disabled or no user identified. There’s nothing to forget!"
        elif "tell me a joke" in command.lower():
            return "Why don’t skeletons fight each other? They don’t have the guts! Anything else?"
        else:
            return f"Grok here, processing '{command}'. Today’s {current_date}, and I’m at your service. What’s on your mind? Need a deeper dive or a witty quip?"

    def handle_unknown_command(self, command):
        self.speak("I don’t recognize that command. Would you like me to learn it or pass it to Grok?")
        response = self.listen()
        if response and "yes" in response.lower():
            self.speak("Please explain what I should do.")
            explanation = self.listen()
            if explanation:
                self.memory["learned_commands"][command] = explanation
                self.save_memory()
                self.speak(f"Command '{command}' learned, {self.get_user_prefs()['greeting']}.")
        elif response and "grok" in response.lower():
            self.speak("Passing to Grok...")
            grok_response = self.get_grok_response(command)
            self.speak(grok_response)

    def system_monitor(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            self.sys_var.set(f"System: CPU {cpu:.1f}% | Mem {mem:.1f}%")
            if cpu > 90 or mem > 90:
                self.speak(f"System alert: CPU at {cpu}%, Memory at {mem}%, {self.get_user_prefs()['greeting']}.")
            time.sleep(5)  # Update every 5 seconds

    def start_threads(self):
        threading.Thread(target=self.system_monitor, daemon=True).start()
        threading.Thread(target=self.wake_word_listener, daemon=True).start()
        for reminder in self.memory["reminders"]:
            reminder_time = datetime.fromisoformat(reminder["time"])
            if reminder_time > datetime.now():
                threading.Thread(target=self.check_reminder, args=(reminder["text"], reminder_time, reminder["user"]), daemon=True).start()

    def listen_loop(self):
        self.speak(f"Manual listening mode activated, {self.get_user_prefs()['greeting']}. How may I assist you?")
        while self.running and self.listening:
            command = self.listen()
            if command:
                self.process_command(command)
            time.sleep(1)
        if self.running:
            self.status_var.set(f"Status: Waiting for '{self.config['wake_word']}'")

    def shutdown(self):
        self.speak(f"Shutting down, {self.get_user_prefs()['greeting']}. Goodbye from MOON and Grok.")
        self.running = False
        pygame.quit()
        self.save_memory()
        for process in self.processes.values():
            if process:
                process.terminate()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    moon = MOON()
    moon.root.mainloop()