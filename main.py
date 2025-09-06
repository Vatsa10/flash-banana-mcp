import os
import sys
import tempfile
import pathlib
import base64
import logging
import json
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Third-party imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, Response
import google.generativeai as genai
from dotenv import load_dotenv
import aiohttp
import io
from PIL import Image

# Debug info
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Flash Banana - Image and Text Processing with Gemini 2.5 Flash")

# Debug template directory
template_dir = Path("templates")
print(f"Template directory exists: {template_dir.exists()}")
print(f"Template directory contents: {list(template_dir.glob('*')) if template_dir.exists() else 'N/A'}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
try:
    templates = Jinja2Templates(directory=str(template_dir))
    print("Templates initialized successfully")
except Exception as e:
    print(f"Error initializing templates: {e}")
    raise

# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment variables")
    # Don't raise error here to allow the app to start without the key

# Initialize Gemini model
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_image_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={
            'temperature': 0.7,
            'max_output_tokens': 2048,
        }
    )
    print("✅ Gemini model initialized successfully")
else:
    print("⚠️ Gemini model not initialized - API key missing")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

async def download_image(image_url: str) -> tuple[bytes, str]:
    """Download image from URL with timeout and error handling"""
    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=timeout, 
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}) as response:
                if not response.ok:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download image: {response.status} {response.reason}"
                    )
                
                content_type = response.headers.get('content-type', 'image/jpeg')
                image_data = await response.read()
                
                if not image_data:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Downloaded image is empty"
                    )
                
                return image_data, content_type
                
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading image: {str(e)}"
        )

def process_image_with_pil(image_data: bytes) -> bytes:
    """Process image with PIL (similar to Sharp in the original code)"""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Get original image info
        print(f"🖼️ Processing image: {img.format} {img.size[0]}x{img.size[1]} {img.mode}")
        
        # Convert to RGB if needed (for PNG with transparency)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        
        # Convert to JPEG with high quality
        output = io.BytesIO()
        img.convert('RGB').save(output, format='JPEG', quality=95, progressive=True, optimize=True)
        
        return output.getvalue()
        
    except Exception as e:
        print(f"❌ PIL processing error: {e}")
        return image_data  # Return original if processing fails

@app.post("/api/process")
async def process_image(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    prompt: str = Form(...)
):
    """
    Process an image with a text prompt using Gemini 2.5 Flash
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key not configured"
        )
    
    try:
        print('🎨 Executing edit with Google Gemini 2.5 Flash Image...')
        print('Change Summary:', prompt)
        
        # Handle either file upload or URL
        if image:
            print('📥 Processing uploaded file...')
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
            image_data = await image.read()
            content_type = image.content_type
            print(f'📄 Uploaded file: {image.filename}, size: {len(image_data)} bytes, type: {content_type}')
            
        elif image_url:
            print(f'📥 Downloading image from URL: {image_url}')
            image_data, content_type = await download_image(image_url)
            print(f'✅ Downloaded image, size: {len(image_data)} bytes, type: {content_type}')
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either image file or image_url must be provided"
            )
        
        # Process image with PIL (similar to Sharp in the original code)
        processed_image_data = process_image_with_pil(image_data)
        print(f'✅ Processed image, size: {len(processed_image_data)} bytes')
        
        # Convert to base64 for Gemini
        image_base64 = base64.b64encode(processed_image_data).decode('utf-8')
        
        # Call Gemini API
        print('🤖 Sending to Google Gemini 2.5 Flash...')
        print('📋 Prompt:', prompt)
        
        try:
            response = await gemini_image_model.generate_content_async([
                prompt,
                {
                    'mime_type': content_type,
                    'data': image_base64
                }
            ])
            
            print('✅ Received response from Google Gemini 2.5 Flash')
            
            # Process the response
            if not response.text:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No response from Gemini API"
                )
            
            # For simplicity, we're just returning the text response
            # In a real app, you'd want to handle different response types
            return {
                "ok": True,
                "result": response.text,
                "method": "google_gemini",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f'❌ Gemini API error: {str(e)}')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calling Gemini API: {str(e)}"
            )
            
    except HTTPException:
        raise
        
    except Exception as e:
        print(f'❌ Unexpected error: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
