from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import os
from filehandler import FileHandler
from producer import Producer
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_file_directory = os.getenv("UPLOAD_DIR")
processed_file_directory = os.getenv("PROCESSED_DIR")
saved_file_directory =  os.getenv("SAVED_DIR")
file_handler = FileHandler(uploaded_file_directory, processed_file_directory)
producer = Producer()

@app.get("/files")
async def get_files():
    return JSONResponse(file_handler.list_files(producer))

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = os.path.join(uploaded_file_directory, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.post("/files")
async def put_files(files: List[UploadFile] = File(...)):
    saved_files = await file_handler.handle_multiple_files(files)
    return {"uploaded": saved_files}

# TODO
@app.delete("files/{file_name}")
async def delete_file(file_name: str):
    pass

@app.get("/show-file/{file_name}")
async def show_editor(file_name: str):
    image_masks = []
    base_image = ""
    edited_image = ""
    try:
        base_image = file_handler.fetch_image(file_name)
        image_masks = file_handler.return_image_masks(file_name)
    except Exception as e:
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        edited_image = file_handler.fetch_image(file_name, True)
    except Exception as e:
        print(e)
        pass

    return {"masks":image_masks, "base_image":base_image, "edited_image":edited_image}

@app.post("/process-files")
async def process_files(request: Request):
    try:
        body = await request.json()
        files = body.get('files')
        prompt = body.get('prompt')
        positive = body.get('positive')
        highlight = body.get('highlight')

        stored_files = list(
            map(
                lambda file: file_handler.file_table.get_stored_name(file, "admin"),
                files
        ))

        producer.create_file_queue(stored_files, prompt, positive, highlight)

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
        return JSONResponse({
            "data": file_handler.download(file_name)
        })

    except:
        JSONResponse({"error": "Download Failed"}, status_code=500)

#@app.on_event("startup")
#def startup_event():
#    init_db()
#    setup_rabbitmq_producer()

@app.on_event("shutdown")
def shutdown_event():
    if producer.channel:
        producer.channel.close()

@app.get("/health")
def health():
    return {"status": "ok"}