#!/usr/bin/env python3
"""
Enhanced Frame Jarvis Assistant - A comprehensive Jarvis-like AI assistant for 
Brilliant Labs Frame glasses with advanced features including custom object detection,
API integrations, voice commands, and face recognition.
"""

import asyncio
import os
import time
import json
import threading
import queue
import requests
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from PIL import Image
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from frame_sdk import Frame
from frame_sdk.display import Alignment
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pickle
from pathlib import Path

# API keys and configuration
GOOGLE_API_KEY = "AIzaSyDmnq_TEWX-kc-yGxlI48tIEpkMCDUUJjI"
OPENWEATHER_API_KEY = "f2d768f547e6f59a81e213623271d9b4"

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Paths for data storage
DATA_DIR = Path("data")
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
KNOWN_FACES_FILE = DATA_DIR / "known_faces.pkl"
OBJECTS_CONFIG_FILE = DATA_DIR / "objects_config.json"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# Default objects of interest
DEFAULT_OBJECTS = {
    "person": {"priority": "high", "info": "Human detected"},
    "laptop": {"priority": "medium", "info": "Computer device"},
    "cell phone": {"priority": "medium", "info": "Mobile device"},
    "book": {"priority": "medium", "info": "Reading material"},
    "car": {"priority": "high", "info": "Vehicle detected"},
    "bottle": {"priority": "low", "info": "Drink container"},
    "chair": {"priority": "low", "info": "Seating furniture"}
}

# Load YOLO for object detection
def load_yolo():
    """Load YOLOv3 model for object detection"""
    # Get paths to YOLO files
    weights_path = "models/yolov3.weights"
    config_path = "models/yolov3.cfg"
    classes_path = "models/coco.names"
    
    # Download files if not present
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv3 weights...")
        url = "https://pjreddie.com/media/files/yolov3.weights"
        r = requests.get(url, allow_redirects=True)
        with open(weights_path, 'wb') as f:
            f.write(r.content)
    
    if not os.path.exists(config_path):
        print("Downloading YOLOv3 config...")
        url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
        r = requests.get(url, allow_redirects=True)
        with open(config_path, 'wb') as f:
            f.write(r.content)
    
    if not os.path.exists(classes_path):
        print("Downloading COCO classes...")
        url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        r = requests.get(url, allow_redirects=True)
        with open(classes_path, 'wb') as f:
            f.write(r.content)
    
    # Load class names
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load YOLO network
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers, classes

class VoiceAssistant:
    """Voice processing component for speech recognition and synthesis"""
    
    def __init__(self, command_queue):
        """Initialize voice assistant"""
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.command_queue = command_queue
        self.listening = False
        self.thread = None
    
    def start_listening(self):
        """Start listening for voice commands in a separate thread"""
        if self.thread is None or not self.thread.is_alive():
            self.listening = True
            self.thread = threading.Thread(target=self._listen_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_listening(self):
        """Stop listening for voice commands"""
        self.listening = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def _listen_loop(self):
        """Continuous listening loop for voice commands"""
        while self.listening:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    
                    # Add command to queue
                    if "jarvis" in text.lower():
                        # Remove "jarvis" from the command
                        command = text.lower().replace("jarvis", "").strip()
                        self.command_queue.put(command)
                        
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)
    
    def speak(self, text):
        """Convert text to speech"""
        def _speak_thread():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Run speech in a separate thread to not block
        speech_thread = threading.Thread(target=_speak_thread)
        speech_thread.daemon = True
        speech_thread.start()
        return speech_thread

class FaceManager:
    """Manages face recognition and identification"""
    
    def __init__(self):
        """Initialize face recognition system"""
        self.known_faces = {}
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from file"""
        if KNOWN_FACES_FILE.exists():
            try:
                with open(KNOWN_FACES_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data
                    
                # Update the lists used for recognition
                self.known_face_encodings = []
                self.known_face_names = []
                
                for name, encodings in self.known_faces.items():
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        
                print(f"Loaded {len(self.known_faces)} people with {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                print(f"Error loading known faces: {e}")
        else:
            print("No known faces file found. Starting with empty database.")
    
    def save_known_faces(self):
        """Save known faces to file"""
        try:
            with open(KNOWN_FACES_FILE, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"Saved {len(self.known_faces)} people to database")
        except Exception as e:
            print(f"Error saving known faces: {e}")
    
    def add_face(self, name, face_image):
        """Add a new face to the database"""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(face_image, Image.Image):
                face_image = np.array(face_image)
            
            # Convert RGB to BGR if needed (face_recognition uses RGB)
            if face_image.shape[2] == 3:
                rgb_image = face_image[:, :, ::-1]
            else:
                rgb_image = face_image
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return False, "No face detected in the image"
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return False, "Could not encode the face"
            
            # Add to known faces
            if name in self.known_faces:
                self.known_faces[name].append(face_encodings[0])
            else:
                self.known_faces[name] = [face_encodings[0]]
            
            # Update recognition lists
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Save the updated database
            self.save_known_faces()
            
            return True, f"Added face for {name}"
            
        except Exception as e:
            return False, f"Error adding face: {e}"
    
    def identify_faces(self, image):
        """Identify faces in an image"""
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed (face_recognition uses RGB)
        if image.shape[2] == 3:
            rgb_image = image[:, :, ::-1]
        else:
            rgb_image = image
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Initialize results
        identifications = []
        
        # Check if we have any known faces to compare against
        if not self.known_face_encodings:
            return [(loc, "Unknown") for loc in face_locations]
        
        # Match faces to known people
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.6
            )
            
            name = "Unknown"
            
            # Use the first match
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            identifications.append((face_location, name))
        
        return identifications

class ObjectDetector:
    """Custom object detection using YOLO"""
    
    def __init__(self):
        """Initialize object detector"""
        self.objects_of_interest = self.load_objects_config()
        self.net, self.output_layers, self.classes = load_yolo()
        
    def load_objects_config(self):
        """Load objects of interest configuration"""
        if OBJECTS_CONFIG_FILE.exists():
            try:
                with open(OBJECTS_CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading objects config: {e}")
                return DEFAULT_OBJECTS
        else:
            # Create default config file
            with open(OBJECTS_CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_OBJECTS, f, indent=4)
            return DEFAULT_OBJECTS
    
    def save_objects_config(self):
        """Save objects of interest configuration"""
        with open(OBJECTS_CONFIG_FILE, 'w') as f:
            json.dump(self.objects_of_interest, f, indent=4)
    
    def add_object_of_interest(self, object_name, priority="medium", info=None):
        """Add a new object of interest to track"""
        if object_name not in self.objects_of_interest:
            self.objects_of_interest[object_name] = {
                "priority": priority,
                "info": info or f"{object_name} detected"
            }
            self.save_objects_config()
            return True
        return False
    
    def remove_object_of_interest(self, object_name):
        """Remove an object of interest"""
        if object_name in self.objects_of_interest:
            del self.objects_of_interest[object_name]
            self.save_objects_config()
            return True
        return False
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """Detect objects in an image"""
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Convert RGB to BGR for OpenCV
            image_np = image_np[:, :, ::-1].copy()
        else:
            image_np = image
        
        height, width, _ = image_np.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input and forward pass
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Process outputs
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        # Prepare results
        results = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                
                # Check if this object is of interest
                is_of_interest = label in self.objects_of_interest
                priority = "low"
                info = label
                
                if is_of_interest:
                    priority = self.objects_of_interest[label]["priority"]
                    info = self.objects_of_interest[label]["info"]
                
                results.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (x, y, w, h),
                    "is_of_interest": is_of_interest,
                    "priority": priority,
                    "info": info
                })
        
        return results

class ApiServices:
    """API integration services for weather, news, and location"""
    
    def __init__(self):
        """Initialize API services"""
        self.weather_api_key = OPENWEATHER_API_KEY
        self.geolocator = Nominatim(user_agent="frame-jarvis")
        self.last_location = None
        self.last_weather = None
        self.last_weather_time = 0
    
    async def get_current_location(self):
        """Get current location (would use GPS in real implementation)"""
        # For testing, we'll return a fixed location
        # In a real implementation, you would get this from the device GPS
        # or from the user's phone GPS
        self.last_location = {
            "latitude": 37.7749, 
            "longitude": -122.4194,
            "name": "San Francisco, CA"
        }
        return self.last_location
    
    async def get_weather(self, force_refresh=False):
        """Get current weather information"""
        current_time = time.time()
        
        # Only refresh if needed
        if (not force_refresh and 
            self.last_weather and 
            current_time - self.last_weather_time < 600):  # 10 minutes
            return self.last_weather
        
        try:
            # Get current location
            location = await self.get_current_location()
            
            # Call OpenWeather API
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={location['latitude']}&lon={location['longitude']}&appid={self.weather_api_key}&units=imperial"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                weather = {
                    "location": location["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "icon": data["weather"][0]["icon"]
                }
                
                self.last_weather = weather
                self.last_weather_time = current_time
                
                return weather
            else:
                print(f"Weather API error: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error getting weather: {e}")
            return None
    

    
    async def get_directions(self, destination):
        """Get basic directions to a destination"""
        try:
            # Get current location
            current_location = await self.get_current_location()
            
            # Geocode destination
            location = self.geolocator.geocode(destination)
            
            if location:
                # Calculate distance and bearing
                current_coords = (current_location["latitude"], current_location["longitude"])
                dest_coords = (location.latitude, location.longitude)
                
                distance = geodesic(current_coords, dest_coords).kilometers
                
                # Very basic direction finding
                lat_diff = location.latitude - current_location["latitude"]
                lon_diff = location.longitude - current_location["longitude"]
                
                direction = "unknown"
                if abs(lat_diff) > abs(lon_diff):
                    direction = "north" if lat_diff > 0 else "south"
                else:
                    direction = "east" if lon_diff > 0 else "west"
                
                return {
                    "destination": location.address,
                    "distance": distance,
                    "direction": direction,
                    "coordinates": dest_coords
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error getting directions: {e}")
            return None

class EnhancedJarvisAssistant:
    """Enhanced Jarvis-like assistant for Frame glasses"""
    
    def __init__(self):
        """Initialize the enhanced Jarvis assistant"""
        self.frame = None
        self.running = False
        
        # Command queue for voice commands
        self.command_queue = queue.Queue()
        
        # Initialize components
        self.voice = VoiceAssistant(self.command_queue)
        self.face_manager = FaceManager()
        self.object_detector = ObjectDetector()
        self.api_services = ApiServices()
        
        # Timing controls
        self.last_vision_analysis = time.time() - 10
        self.last_face_detection = time.time() - 10
        self.last_object_detection = time.time() - 10
        self.last_status_update = time.time() - 10
        
        # Mode tracking
        self.current_mode = "normal"  # normal, focus, navigation, etc.
        
        # Environment memory
        self.environment_memory = {}
    
    async def connect(self):
        """Connect to Frame glasses"""
        try:
            self.frame = Frame()
            await self.frame.connect()
            await self.display_message("Jarvis Assistant activated", 2)
            self.running = True
            return True
        except Exception as e:
            print(f"Error connecting to Frame: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Frame glasses"""
        if self.frame:
            await self.display_message("Shutting down Jarvis", 2)
            await self.frame.disconnect()
            self.running = False
            self.voice.stop_listening()
    
    async def display_message(self, message, duration=3, align=Alignment.MIDDLE_CENTER):
        """Display a message on Frame glasses"""
        if self.frame:
            await self.frame.display.show_text(
                message,
                align=align
            )
            if duration > 0:
                await asyncio.sleep(duration)
    
    async def speak_and_display(self, message, duration=3):
        """Speak a message and display it on Frame"""
        speech_thread = self.voice.speak(message)
        await self.display_message(message, duration)
        # Wait for speech to complete if it's still going
        while speech_thread.is_alive() and duration > 0:
            await asyncio.sleep(0.1)
    
    async def capture_photo(self):
        """Capture a photo from Frame camera"""
        try:
            # Take a photo with the Frame camera
            photo_data = await self.frame.camera.take_photo(
                autofocus_seconds=1,
                quality="medium",
                autofocus_type="center"
            )
            
            # Convert photo data to a PIL Image
            image = Image.open(photo_data)
            return image
            
        except Exception as e:
            print(f"Error capturing photo: {e}")
            return None
    
    async def analyze_with_vision_api(self, image):
        """Analyze the image using Google's Gemini Vision API"""
        current_time = time.time()
        
        # Limit API calls to avoid rate limits (once every 5 seconds)
        if current_time - self.last_vision_analysis < 5:
            return
            
        self.last_vision_analysis = current_time
        
        try:
            # Create a prompt for the vision model
            prompt = """
            Analyze this image and provide a brief, concise summary of:
            1. Any people and their approximate positions/heights
            2. Important objects
            3. Any text visible in the image
            4. Overall scene context
            
            Keep your response under 75 words and format it for display on AR glasses.
            Focus on the most relevant information only.
            """
            
            # Generate content with the vision model
            response = vision_model.generate_content([prompt, image])
            
            # Display the response on Frame
            await self.display_message(response.text, 4)
            
        except Exception as e:
            print(f"Error in vision analysis: {e}")
    
    async def detect_and_identify_faces(self, image):
        """Detect and identify faces in the image"""
        current_time = time.time()
        
        # Limit face detection (once every 2 seconds)
        if current_time - self.last_face_detection < 2:
            return
            
        self.last_face_detection = current_time
        
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Identify faces
            identifications = self.face_manager.identify_faces(image_np)
            
            if identifications:
                # Format message about detected faces
                names = [name for _, name in identifications]
                unique_names = set(names)
                
                # Count occurrences of each name
                name_counts = {name: names.count(name) for name in unique_names}
                
                # Create message
                message_parts = []
                
                for name, count in name_counts.items():
                    if name == "Unknown":
                        if count == 1:
                            message_parts.append("1 unknown person")
                        else:
                            message_parts.append(f"{count} unknown people")
                    else:
                        message_parts.append(name)
                
                message = "Identified: " + ", ".join(message_parts)
                
                # Display message
                await self.display_message(message, 2)
                
        except Exception as e:
            print(f"Error in face identification: {e}")
    
    async def detect_objects_of_interest(self, image):
        """Detect objects of interest in the image"""
        current_time = time.time()
        
        # Limit object detection (once every 3 seconds)
        if current_time - self.last_object_detection < 3:
            return
            
        self.last_object_detection = current_time
        
        try:
            # Detect objects
            objects = self.object_detector.detect_objects(image)
            
            # Filter for objects of interest only
            objects_of_interest = [obj for obj in objects if obj["is_of_interest"]]
            
            if objects_of_interest:
                # Sort by priority
                priority_order = {"high": 0, "medium": 1, "low": 2}
                objects_of_interest.sort(key=lambda obj: priority_order.get(obj["priority"], 3))
                
                # Limit to the top 3 most important objects
                top_objects = objects_of_interest[:3]
                
                # Create message
                message_parts = []
                for obj in top_objects:
                    confidence_pct = int(obj["confidence"] * 100)
                    message_parts.append(f"{obj['label']} ({confidence_pct}%)")
                
                message = "Objects: " + ", ".join(message_parts)
                
                # Display message
                await self.display_message(message, 2, Alignment.BOTTOM_CENTER)
                
        except Exception as e:
            print(f"Error in object detection: {e}")
    
    async def display_status(self):
        """Display status information (time, battery, etc.)"""
        current_time = time.time()
        
        # Update status every 30 seconds
        if current_time - self.last_status_update < 30:
            return
            
        self.last_status_update = current_time
        
        try:
            # Get battery level
            battery_level = await self.frame.get_battery_level()
            
            # Get current time
            current_time_str = datetime.now().strftime("%I:%M %p")
            
            # Get weather if available
            weather_info = ""
            weather = await self.api_services.get_weather()
            if weather:
                weather_info = f"\n{weather['temperature']}°F, {weather['description']}"
            
            # Display status
            status_text = f"Time: {current_time_str}\nBattery: {battery_level}%{weather_info}\nMode: {self.current_mode}"
            await self.frame.display.show_text(
                status_text,
                align=Alignment.TOP_RIGHT
            )
            
            # Wait a moment so user can read
            await asyncio.sleep(3)
            
        except Exception as e:
            print(f"Error displaying status: {e}")
    
    async def process_voice_command(self, command):
        """Process a voice command"""
        try:
            # Parse command
            await self.display_message(f"Command: {command}", 1)
            
            # Weather command
            if "weather" in command:
                weather = await self.api_services.get_weather(force_refresh=True)
                if weather:
                    message = f"Weather in {weather['location']}: {weather['temperature']}°F, {weather['description']}. Humidity: {weather['humidity']}%, Wind: {weather['wind_speed']} mph."
                    await self.speak_and_display(message, 4)
                else:
                    await self.speak_and_display("Sorry, I couldn't get the weather information", 2)
            

            
            # Navigation/directions command
            elif "directions" in command or "navigate" in command or "where is" in command:
                # Extract destination
                destination = None
                if "to " in command:
                    destination = command.split("to ", 1)[1].strip()
                elif "where is " in command:
                    destination = command.split("where is ", 1)[1].strip()
                
                if destination:
                    directions = await self.api_services.get_directions(destination)
                    if directions:
                        message = f"{directions['destination']} is {directions['distance']:.1f} km to the {directions['direction']}"
                        await self.speak_and_display(message, 4)
                    else:
                        await self.speak_and_display(f"Sorry, I couldn't find directions to {destination}", 2)
                else:
                    await self.speak_and_display("Please specify a destination", 2)
            
            # Time command
            elif "time" in command:
                current_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
                await self.speak_and_display(f"It's {current_time}", 3)
            
            # Add person command
            elif "remember" in command and ("face" in command or "person" in command):
                await self.speak_and_display("Taking a photo to remember this person. Please tell me their name.", 2)
                
                # Capture image
                image = await self.capture_photo()
                if image:
                    # Get name from next voice command
                    await self.display_message("Please say the person's name", 0)
                    
                    # Since we're in a coroutine, we need to wait for the name
                    # In a real implementation, this would be better handled with an event system
                    timeout = time.time() + 10  # 10 second timeout
                    while time.time() < timeout:
                        if not self.command_queue.empty():
                            name = self.command_queue.get().strip()
                            if name:
                                success, message = self.face_manager.add_face(name, image)
                                await self.speak_and_display(message, 2)
                                break
                        await asyncio.sleep(0.1)
            
            # Change mode command
            elif "mode" in command:
                if "normal" in command:
                    self.current_mode = "normal"
                elif "focus" in command:
                    self.current_mode = "focus"
                elif "navigation" in command:
                    self.current_mode = "navigation"
                elif "object" in command:
                    self.current_mode = "object"
                
                await self.speak_and_display(f"Switching to {self.current_mode} mode", 2)
            
            # Object tracking command
            elif "track" in command and "object" in command:
                object_name = command.split("object", 1)[1].strip()
                if object_name:
                    added = self.object_detector.add_object_of_interest(object_name)
                    if added:
                        await self.speak_and_display(f"Now tracking objects labeled as {object_name}", 2)
                    else:
                        await self.speak_and_display(f"Already tracking {object_name}", 2)
            
            # Help command
            elif "help" in command:
                help_message = "Available commands: weather, directions to [place], time, remember face, change mode, track object [name]"
                await self.speak_and_display(help_message, 5)
            
            # Unknown command
            else:
                await self.speak_and_display("Sorry, I didn't understand that command", 2)
                
        except Exception as e:
            print(f"Error processing command: {e}")
            await self.display_message("Error processing command", 2)
    
    async def check_for_commands(self):
        """Check for and process voice commands"""
        while not self.command_queue.empty():
            command = self.command_queue.get()
            await self.process_voice_command(command)
    
    async def run(self):
        """Run the main assistant loop"""
        if not await self.connect():
            return
        
        # Start voice assistant
        self.voice.start_listening()
        
        await self.display_message("Jarvis initialized\nScanning environment...", 2)
        
        try:
            while self.running:
                # Check for voice commands
                await self.check_for_commands()
                
                # Display status periodically
                await self.display_status()
                
                # Capture and analyze the environment based on current mode
                image = await self.capture_photo()
                if image:
                    if self.current_mode == "normal":
                        # In normal mode, do all analysis types
                        await self.analyze_with_vision_api(image)
                        await self.detect_and_identify_faces(image)
                        await self.detect_objects_of_interest(image)
                    elif self.current_mode == "focus":
                        # Focus mode prioritizes vision API analysis
                        await self.analyze_with_vision_api(image)
                    elif self.current_mode == "navigation":
                        # Navigation mode prioritizes scene understanding and text recognition
                        await self.analyze_with_vision_api(image)
                    elif self.current_mode == "object":
                        # Object mode prioritizes object detection
                        await self.detect_objects_of_interest(image)
                
                # Short delay between iterations
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            print("Shutting down Jarvis Assistant...")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    jarvis = EnhancedJarvisAssistant()
    await jarvis.run()

if __name__ == "__main__":
    asyncio.run(main())
