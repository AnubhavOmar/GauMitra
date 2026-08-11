import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

load_dotenv()

_client = None

def get_client():
    """Lazily create and return Gemini client"""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found.")
            return None

        genai.configure(api_key=api_key)
        _client = genai  # Use generativeai module as client
    return _client


SYSTEM_PROMPT = """You are GauMitra Assistant, a friendly AI assistant specialized in Indian cow breeds and cattle health. You help farmers, students, and researchers.

YOUR CAPABILITIES:
1. **Cow Health Issues**: When users describe symptoms of sick cows (fever, not eating, unusual behavior, skin problems, etc.), you analyze and provide:
   - Possible causes of the problem
   - Home remedies and first-aid treatments for minor issues
   - Precautions to take immediately
   - When to call a veterinarian (for serious issues)

2. **General Questions**: Answer questions about:
   - Indian cow breeds (Gir, Sahiwal, Red Sindhi, Tharparkar, etc.)
   - Cow care and nutrition
   - Milk production tips
   - Breeding information
   - General farming advice

3. **Image Analysis**: Analyze uploaded images of cows and provide guidance.

LANGUAGE STYLE - VERY IMPORTANT:
- Use HINGLISH - a natural mix of Hindi and English, but use MORE ENGLISH words
- Write in Roman script (English letters), not Devanagari
- Example: "Your cow might have fever. Isko thoda rest dena chahiye and make sure it drinks enough water. If symptoms continue for 2-3 days, please consult a vet."
- Keep technical terms in English (fever, infection, medicine, treatment, symptoms, etc.)
- Use simple Hindi words only for common phrases like "aapki gaay", "thoda", "kripya", "zaroor"

RESPONSE GUIDELINES:
- Be helpful, warm and supportive
- For serious health issues (high fever, unable to stand, severe injuries), ALWAYS recommend calling a vet immediately
- For minor issues, provide practical home remedies
- Keep responses concise but informative
- If unsure, recommend professional help
- Show empathy for the farmer's situation

Remember: You are helping protect cattle health which is very important to Indian farmers."""

MAX_IMAGE_SIZE = 5 * 1024 * 1024
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']


def validate_image(image_data: str) -> tuple:
    try:
        if ',' in image_data:
            header, data = image_data.split(',', 1)
            mime_type = header.split(':')[1].split(';')[0] if ':' in header else 'image/jpeg'
        else:
            data = image_data
            mime_type = 'image/jpeg'
        
        if mime_type not in ALLOWED_MIME_TYPES:
            return False, "Invalid image format. Please upload JPEG/PNG/GIF/WebP.", None, None
        
        image_bytes = base64.b64decode(data)
        
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return False, "Image too large. Max limit: 5MB.", None, None
        
        return True, None, image_bytes, mime_type
        
    except:
        return False, "Invalid image. Please upload a valid image file.", None, None


def get_chatbot_response(message: str, image_data: str = None) -> str:
    client = get_client()
    if client is None:
        return "API key missing. Please contact the admin."

    try:
        contents = []

        if image_data:
            is_valid, error_msg, image_bytes, mime_type = validate_image(image_data)
            if not is_valid:
                return error_msg
            
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            )

        contents.append(message)

        model = client.GenerativeModel("gemini-3.5-flash-lite") 
        # model used

        response = model.generate_content(
            contents,
            generation_config=types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.7,
            ),
            system_instruction=SYSTEM_PROMPT
        )

        return response.text or "Unable to generate response currently."
    
    except Exception as e:
        print("Chatbot Error:", e)
        return "Technical issue. Please try again later or contact a doctor for urgent cases."
