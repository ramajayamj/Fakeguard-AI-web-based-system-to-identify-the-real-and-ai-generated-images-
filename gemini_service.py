import os
import google.generativeai as genai
import json

# ==========================================
# CONFIGURATION
# ==========================================
# HARDCODED API KEY AS REQUESTED. 
# PLEASE REPLACE 'YOUR_API_KEY_HERE' WITH YOUR ACTUAL GENAI API KEY.
API_KEY = "AIzaSyBuugZ-4moVMuoVhkQdDL2lOFzMEF9FePo"

# Using 1.5 Flash as it is the current fast/stable model. 
# If "2.5" works in the future, just update this string.
MODEL_NAME = "gemini-2.5-flash" 

def configure_genai():
    """Configures the Gemini API with the provided key."""
    if API_KEY == "YOUR_API_KEY_HERE":
        print("WARNING: Gemini API Key is not set in gemini_service.py. Please replace the placeholder.")
    genai.configure(api_key=API_KEY)

def analyze_content(file_path_or_bytes, mime_type):
    """
    Analyzes an image or video to determine if it's Real or Fake.
    Returns: (class_name, confidence_string, confidence_float)
    """
    try:
        configure_genai()
        model = genai.GenerativeModel(MODEL_NAME)

        # Upload the file or create a data part
        if isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes):
            # It's a file path (video or temp image)
            # For videos, we might need to upload using the File API for better handling
            # But for simplicity/speed in prototype, we can try passing the file directly if supported
            # or upload it. The File API is safer for videos.
            print(f"Uploading file: {file_path_or_bytes}...")
            uploaded_file = genai.upload_file(file_path_or_bytes, mime_type=mime_type)
            content = [uploaded_file]
        else:
            # It's raw bytes (image)
            content = [{
                "mime_type": mime_type,
                "data": file_path_or_bytes
            }]

        prompt = """
        You are an expert AI Fake Image/Video Detector for an instant delivery refund system.
        Analyze the provided image/video evidence to determine if it is "Real" (genuine product issue) or "Fake" (AI-generated, edited, or reused/fraudulent).
        
        Look for:
        - AI generation artifacts (unnatural lighting, texture glitches, text errors)
        - Editing traces (blurring, mismatched noise, compression artifacts)
        - Recycling/Screenshot evidence.
        
        Return ONLY a JSON object with this format (no markdown):
        {
            "class": "Real" or "Fake",
            "confidence": <number between 0.0 and 1.0>
        }
        """

        response = model.generate_content([prompt] + content)
        
        # Parse JSON
        text = response.text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()
        
        result = json.loads(text)
        
        cls = result.get("class", "Real") # Default to Real if unsure? Or Fake? Payer safe.
        conf = float(result.get("confidence", 0.5))
        
        return cls, f"{conf:.2%}", conf

    except Exception as e:
        print(f"Gemini Analysis Error: {e}")
        # Fallback
        return "Real", "0.00%", 0.0

def chat_response(history, user_message):
    """
    Generates a chat response using Gemini.
    history: list of {"role": "user"|"model", "parts": ["text"]}
    """
    try:
        configure_genai()
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Convert simple history to Gemini format if needed, or maintain state in frontend/app
        # For this simple prototype, we'll just send the current message + minimal context instructions
        
        chat = model.start_chat(history=history or [])
        
        system_instruction = """
        You are a helpful Refund Support Bot for a delivery app.
        Your goal is to guide users to submit valid evidence (images/videos) for their refund claims.
        Be concise, professional, and helpful.
        If they talk about 'fake' or 'real' images, explain that our AI checks for authenticity.
        """
        
        response = chat.send_message(system_instruction + "\nUser: " + user_message)
        return response.text
        
    except Exception as e:
        print(f"Gemini Chat Error: {e}")
        return "I'm having trouble connecting to the support server right now. Please try again."
