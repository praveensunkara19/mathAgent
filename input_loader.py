from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import easyocr
import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2

load_dotenv()



def upload_file(pdf_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        temp_path = tmp.name

    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()

    text = "\n".join([doc.page_content for doc in docs])

    return text



reader = easyocr.Reader(['en'])

def upload_image(image):

    pil_image = Image.open(image)

    img = np.array(pil_image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result = reader.readtext(
        img,
        detail=0,
        paragraph=True
    )

    text = "\n".join(result)

    return text



elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def record_voice(buffer):

    result = elevenlabs.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1",
        tag_audio_events=False,
        language_code="eng",
        diarize=False
    )

    return result.text