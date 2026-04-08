from fastapi import FastAPI, UploadFile, File, Request, Depends # type: ignore
from fastapi.responses import JSONResponse, StreamingResponse # type: ignore
from typing import List
import os
from filehandler import FileHandler
from producer import Producer
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pathlib import Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # type: ignore
from userManager import UserManager
from jwt.exceptions import DecodeError # type: ignore
from requestTypes import LoginRequest, UpdateUserRequest, DownloadRequest, ShowFilesRequest, ProcessRequest, DeleteFile, ShareFileRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_file_directory = os.getenv("UPLOAD_DIR")
processed_file_directory = os.getenv("PROCESSED_DIR")
saved_file_directory =  os.getenv("SAVED_DIR")
file_handler = FileHandler(uploaded_file_directory, processed_file_directory)
producer = Producer()
user_manager = UserManager()
security = HTTPBearer()

@app.get("/files")
async def get_files(credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        return JSONResponse(file_handler.list_files(user_manager.get_user_id(user), producer))
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/files")
async def put_files(files: List[UploadFile] = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        saved_files = await file_handler.handle_multiple_files(user_manager.get_user_id(user), files)
        return {"uploaded": saved_files}
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/delete-file")
async def delete_file(request: DeleteFile, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        file_handler.delete_file(user_manager.get_user_id(user), request.file_name, request.file_status)
        return JSONResponse({"message": "Deleted Successfully"}, status_code=200)
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/show-file")
async def show_editor(request: ShowFilesRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        image_masks = []
        base_image = ""
        edited_image = ""
        try:
            if request.saved:
                edited_image = file_handler.fetch_saved_image(user_manager.get_user_id(user), request.file_name)
            else:
                edited_image = file_handler.fetch_processed_image(user_manager.get_user_id(user), request.file_name, True)

            base_image = file_handler.fetch_processed_image(user_manager.get_user_id(user), request.file_name)
            image_masks = file_handler.return_image_masks(user_manager.get_user_id(user), request.file_name)

        except Exception as e:
            return JSONResponse({"error": "File not found"}, status_code=404)

        return {"masks":image_masks, "base_image":base_image, "edited_image":edited_image}
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.get("/view-upload-file/{file_name}")
async def show_editor(file_name: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        base_image = ""
        try:
            base_image = file_handler.fetch_uploaded_image(user_manager.get_user_id(user), file_name)
        except Exception as e:
            return JSONResponse({"error": "File not found"}, status_code=404)

        return {"base_image":base_image}
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/process-files")
async def process_files(request: ProcessRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        try:
            user_id = user_manager.get_user_id(user)
            stored_files = list(
                map(
                    lambda file: file_handler.file_table.get_stored_name(file, user_id),
                    request.files
            )) 

            producer.create_file_queue(
                user_id,
                stored_files,
                request.prompt,
                request.positive,
                request.highlight,
                request.saved
            )
            return {"message":"success"}
        except Exception as e:
            JSONResponse({"error": "Processing Failed"}, status_code=500)
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/save-image/{file_name}")
async def save_edited_file(file_name: str, request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        try:
            body = await request.json()
            file = body.get("file")
            file_type = body.get("file_type")
            file_handler.save(user_manager.get_user_id(user), file, file_name, file_type)
            return {"message": "Saved Successfully"}
        except:
            JSONResponse({"error": "Processing Failed"}, status_code=500)
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.get("/download/{file_name}")
async def download_file(file_name: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        try:
            return JSONResponse({
                "data": file_handler.download(user_manager.get_user_id(user), file_name)
            })

        except:
            JSONResponse({"error": "Download Failed"}, status_code=500)
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)


@app.on_event("shutdown")
def shutdown_event():
    if producer.channel:
        producer.channel.close()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/create-user")
async def create_user(request: LoginRequest):
    try:
        user_info = user_manager.create_user(request.username, request.password)
        if user_info:
            return {"data": user_info}
        else:
            JSONResponse({"error": "Username already in use"}, status_code=409)
    except Exception as e:
        JSONResponse({"error": "User Create Failed"}, status_code=500)

@app.post("/delete-user")
async def delete_user(request: LoginRequest):
    try:
        user_manager.delete_user(request.username, request.password)
        return JSONResponse({"error": "Deleted Successfully"}, status_code=204)
    except Exception as e:
        JSONResponse({"error": "Delete User Failed"}, status_code=500)

@app.put("/update-username")
async def update_username(request: UpdateUserRequest):
    try:
        user_info = user_manager.update_user_name(request.username, request.newUsername, request.password)
        if user_info:
            return {"data": user_info}
    except Exception as e:
        JSONResponse({"error": "Update Failed"}, status_code=500)

@app.put("/update-password")
async def update_password(request: UpdateUserRequest):
    try:
        user_info = user_manager.update_user_password(request.username, request.password, request.newPassword)
        if user_info:
            return {"data": user_info}
    except Exception as e:
        JSONResponse({"error": "Update Failed"}, status_code=500)

@app.post("/login")
async def login(request: LoginRequest):
    access_token = user_manager.authenticate_user(request.username, request.password)
    if access_token:
        return JSONResponse({"data":access_token}, status_code=200)
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/download-zip")
async def download_zip(request: DownloadRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):

    user = authenticate_token(credentials)
    if user:
        try:
            return StreamingResponse(
                file_handler.download_zip_file(user_manager.get_user_id(user), request.files),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={request.name}.zip"}
            )
        except Exception as e:
            return JSONResponse({"error":"Error Generating Zip"}, status_code=500)     
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/download-tar")
async def download_tar(request: DownloadRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = authenticate_token(credentials)
    if user:
        try:
            return StreamingResponse(
                file_handler.download_tar_file(user_manager.get_user_id(user), request.files),
                media_type="application/x-tar",
                headers={"Content-Disposition": f"attachment; filename={request.name}.tar.gz"}
            )
        except Exception as e:
            return JSONResponse({"error":"Error Generating Tar"}, status_code=500)     
    else:
        return JSONResponse({"error":"Access denied"}, status_code=401)

@app.post("/share-files")
async def share_files(request: ShareFileRequest,  credentials: HTTPAuthorizationCredentials = Depends(security)):
    pass

def authenticate_token(credentials) -> str | None:
    try:
        user = user_manager.verify_user_token(credentials.credentials)
        return user

    except DecodeError as error:
        return None
