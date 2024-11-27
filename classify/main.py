import asyncio
import time
import logging
import os
import sys

from tempfile import SpooledTemporaryFile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import clip
from clip import load
from starlette.status import HTTP_504_GATEWAY_TIMEOUT, HTTP_429_TOO_MANY_REQUESTS
from prometheus_fastapi_instrumentator import Instrumentator

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

# Image transformation pipeline
transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

REQUEST_TIMEOUT_ERROR = 5  # Threshold

# Load ML Models at Server Init time
# Also augment with prometheus
# Without this GPU memory is allocatd in multiple processes
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML Models")
    # load the ML Model
    model, preprocess = clip.load("ViT-B/32", device=device)

    instrumentator.expose(app)

    # TODO make a type-safe holder
    app.model = model
    yield
    logger.info("Unloading ML Models")

app = FastAPI(name="Image Classify", lifespan=lifespan)
instrumentator = Instrumentator().instrument(app)

# Return a 504 error if the request processing time is above a certain threshold
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_ERROR)

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        return JSONResponse({'detail': 'Request processing time excedeed limit',
                             'processing_time': process_time},
                            status_code=HTTP_504_GATEWAY_TIMEOUT)


@app.post('/classify_image/')
def classify_image(category: list[str] = Query(1), upload: UploadFile = File(...)):
    # processes categories as a query param
    categories = category
    logger.info(f"categories: {categories}")

    raw = upload.file.read()  # this is dangerous for big files
    image = Image.open(BytesIO(raw)).convert('RGB')
    
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # Prepare the text inputs
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
    
    # Compute the features and compare the image to the text inputs
    with torch.no_grad():
        image_features = app.model.encode_image(image)
        text_features = app.model.encode_text(text)
        
    # Compute the raw similarity score
    similarity = (100.0 * image_features @ text_features.T)

    similarity_softmax = similarity.softmax(dim=-1)
    
    # Define a threshold
    threshold = 10.0

    # Get the highest scoring category
    max_raw_score = torch.max(similarity)

    if max_raw_score < threshold:
        return {
            "file_size": len(raw), 
            "category": "none", 
            "similarity_score": 0,
            "values": [0.0 for _ in categories]
        }
    else:
        category_index = similarity_softmax[0].argmax().item()
        category = categories[category_index]
        similarity_score = similarity_softmax[0, category_index].item()
        values = similarity[0].tolist()
        return {
            "file_size": len(raw), 
            "category": category, 
            "similarity_score": similarity_score,
            "values": values
        }

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }
