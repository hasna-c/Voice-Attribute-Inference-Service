import uuid
import time
import io
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from app.audio_processor import AudioProcessor

app = FastAPI()
processor = AudioProcessor()

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    contact_id: str = Form(None)
):
    start_time = time.time()
    
    if not contact_id:
        contact_id = str(uuid.uuid4())
        
    try:
        audio_bytes = await file.read()
        
        # 1. Primary Decoding via soundfile
        try:
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                sr = 16000
        # 2. Fallback Decoding via librosa
        except Exception:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            
    except Exception as e:
        print(f"Audio loading failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio format or corrupted file: {str(e)}")

    # Quality check & Inference
    quality = processor.assess_quality(audio_data)
    gender, age_bracket, language = processor.predict_attributes(audio_data, sr, quality)
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    return {
        "contact_id": contact_id,
        "gender": gender,
        "age_bracket": age_bracket,
        "language": language,
        "processing_ms": processing_ms,
        "audio_quality": quality
    }