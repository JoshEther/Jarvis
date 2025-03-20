# Enhanced Frame Jarvis Assistant

A comprehensive Jarvis-like AI assistant for Brilliant Labs Frame smart glasses with advanced capabilities including custom object detection, API integrations, voice commands, and sophisticated face recognition.

## Overview

This application transforms your Brilliant Labs Frame glasses into a powerful AI assistant similar to Jarvis from Iron Man. It captures images from the Frame's camera, processes them using various AI technologies, and provides contextual information directly in your line of sight.

## Features

### 1. Custom Object Detection
- Real-time detection of objects using YOLOv3
- Configurable objects of interest with priority levels
- Focus on objects that matter to you
- Detailed information about detected objects

### 2. API Integrations
- **Weather**: Real-time weather information for your location
- **Navigation**: Basic directions and distance information
- **Location**: Geographic awareness (note: uses simulated location data by default)

### 3. Voice Command System
- Natural language voice control
- Wake word detection ("Jarvis")
- Command processing for various functions
- Voice responses via text-to-speech

### 4. Advanced Face Recognition
- Face detection and identification
- Database of known faces
- Add new faces on the fly
- Track and identify people in your field of view

### 5. Multiple Operation Modes
- **Normal Mode**: All features active
- **Focus Mode**: Prioritizes vision API analysis
- **Navigation Mode**: Emphasizes scene understanding
- **Object Mode**: Concentrates on object detection

## Requirements

- Brilliant Labs Frame smart glasses
- Python 3.7+
- Google Gemini API key
- OpenWeather API key (for weather features)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/enhanced-frame-jarvis.git
   cd enhanced-frame-jarvis
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Google Gemini API key is already included in the code
   - OpenWeather API key is already included in the code

4. Create necessary directories:
   ```
   mkdir -p data/known_faces models
   ```

## Usage

1. Make sure your Frame glasses are charged and turned on
2. Pair your Frame glasses with your computer via Bluetooth
3. Run the application:
   ```
   python enhanced-frame-jarvis-app.py
   ```
4. The assistant will connect to your Frame glasses and begin the initialization process
5. Speak commands prefixed with "Jarvis" (e.g., "Jarvis, what's the weather?")

## Voice Commands

The assistant responds to the following voice commands (always prefixed with "Jarvis"):

- **Weather**: "Jarvis, what's the weather?" or "Jarvis, weather forecast"
- **Directions**: "Jarvis, directions to Central Park" or "Jarvis, where is the library"
- **Time**: "Jarvis, what time is it?"
- **Remember Face**: "Jarvis, remember this face" or "Jarvis, remember this person"
- **Change Mode**: "Jarvis, normal mode" or "Jarvis, focus mode" or "Jarvis, navigation mode" or "Jarvis, object mode"
- **Object Tracking**: "Jarvis, track object coffee cup"
- **Help**: "Jarvis, help" - Lists available commands

## How It Works

The application uses a modular architecture with several key components:

1. **VoiceAssistant**: Handles speech recognition and text-to-speech
2. **FaceManager**: Manages face recognition and identification
3. **ObjectDetector**: Detects and tracks objects using YOLO
4. **ApiServices**: Handles weather and location services
5. **EnhancedJarvisAssistant**: Core class that coordinates all components

The system captures images from the Frame's camera and processes them using:
- Google Gemini Vision API for comprehensive scene understanding
- Local face recognition using the face_recognition library
- YOLO object detection for real-time object identification

Results are displayed directly on the Frame's display, providing contextual augmented reality information.

## Customization

You can customize the application by:

- Adding new objects to track in the `objects_config.json` file
- Adding face images to the known faces database
- Modifying the prompts sent to Gemini Vision API
- Creating new voice commands and handlers
- Adjusting timing parameters for various analyses

## Privacy Note

This application processes images taken by your Frame glasses. The images are sent to Google's Gemini API for analysis but are not permanently stored. Face recognition data is stored locally. Be mindful of privacy considerations when using this in public spaces.

## Future Enhancements

- Emotion recognition for detected faces
- Advanced spatial mapping and memory
- Integration with smart home systems
- Calendar and task management
- Multi-language support
- Gesture recognition

## License

MIT License

## Acknowledgments

- Brilliant Labs for creating the innovative Frame smart glasses
- Google for the Gemini Vision API
- OpenCV and YOLO developers for computer vision tools
- The face_recognition library developers

---

For support or questions, please open an issue on this repository or contact [your contact information].
