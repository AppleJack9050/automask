from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import os
from filehandler import FileHandler
from fastapi.middleware.cors import CORSMiddleware
from fileProcessingHandler import FileProcessor
from pathlib import Path
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### COMMENT OUT WHEN NOT USING A MAC
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

file_directory = "./uploads"
processed_file_directory = "./processed"
saved_file_directory = "./saved"
file_handler = FileHandler(file_directory, processed_file_directory)
file_processor = FileProcessor(file_directory, processed_file_directory)

os.makedirs(file_directory, exist_ok=True)

@app.get("/files")
async def get_files():
    unprocessed_files = os.listdir(file_directory)
    processed_files = os.listdir(processed_file_directory)
    saved_files = os.listdir(saved_file_directory)
    files_being_processed = []
    with file_processor.file_q.mutex:
        queue = list(file_processor.file_q.queue)

        files_being_processed = queue

    for file in files_being_processed:
        if file in unprocessed_files:
            unprocessed_files.remove(file)

    return JSONResponse([
        {"Unprocessed": unprocessed_files},
        {"Processing": files_being_processed},
        {"Processed": processed_files},
        {"Saved": saved_files}
    ])

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = os.path.join(file_directory, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.post("/files")
async def put_files(files: List[UploadFile] = File(...)):
    saved_files = await file_handler.handle_multiple_files(files)
    return {"uploaded": saved_files}

@app.get("/show-file/{file_name}")
async def show_editor(file_name: str):
    file_path = os.path.join(processed_file_directory, Path(file_name).stem)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    base_image = file_handler.fetch_image(os.path.join(file_path, "original"), file_name)

    try:
        edited_image = file_handler.fetch_image(os.path.join(file_path, "edited"), file_name)
    except Exception as e:
        pass

    image_masks = file_handler.return_image_masks(file_path)

    return {"masks":image_masks, "base_image":base_image, "edited_image":edited_image}

@app.post("/process-files")
async def process_files(request: Request):
    try:
        body = await request.json()
        files = body.get('files')
        prompt = body.get('prompt')
        positive = body.get('positive')
        highlight = body.get('highlight')

        file_processor.create_process_queue(files)
        await asyncio.to_thread(file_processor.process_files_in_queue(prompt, positive, highlight))
        return {"message":"success"}
    except Exception as e:
        JSONResponse({"error": "Processing Failed"}, status_code=500)

@app.post("/save-image/{file_name}")
async def save_edited_file(file_name: str, request: Request):
    try:
        body = await request.json()
        file = body.get("file")
        file_type = body.get("file_type")
        file_handler.save(file, file_name, file_type)
        return {"message": "Saved Successfully"}
    except:
        JSONResponse({"error": "Processing Failed"}, status_code=500)

@app.get('/download/{file_name}')
async def download_file(file_name: str):
    try:
        file_path = os.path.join(saved_file_directory, file_name)
        if not os.path.exists(file_path):
            return JSONResponse({"error": "File not found"}, status_code=404)

        return JSONResponse({
            "data": file_handler.download(file_path)
        })

    except:
        JSONResponse({"error": "Processing Failed"}, status_code=500)
