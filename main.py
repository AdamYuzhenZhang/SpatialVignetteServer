# main.py
from fastapi import FastAPI, UploadFile, File
from typing import Optional

# test run it
# uvicorn main:app --reload

# 1. Create the FastAPI app object
app = FastAPI()

# 2. Define a "root" endpoint for basic testing
@app.get("/")
def read_root():
    return {"message": "Hello from the Spatial Vignettes Server!"}

# 3. Define the endpoint for processing a vignette
# This simulates receiving a point cloud file from your iOS app
@app.post("/process_vignette/")
async def process_vignette(file: UploadFile = File(...)):
    # In a real app, you would save and process this file
    # For now, we'll just return a confirmation
    print(f"Received file: {file.filename}")

    # This is where your Python abstraction code (PCA, RANSAC, etc.) will go.
    # After processing, you'll return a JSON response.
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "File received, processing would happen here.",
        "tags": ["architecture", "chair", "asymmetry"], # Example tags from LLM
        "asset_urls": {
            "raw": "/path/to/raw.ply",
            "axes": "/path/to/axes.obj",
            "planes": "/path/to/planes.obj"
        }
    }