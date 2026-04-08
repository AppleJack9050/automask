from pydantic import BaseModel # type: ignore
from typing import List

class LoginRequest(BaseModel):
    username: str
    password: str

class UpdateUserRequest(BaseModel):
    username: str
    password: str
    newPassword: str
    newUsername: str

class DownloadRequest(BaseModel):
    files: List[str]
    name: str

class ShowFilesRequest(BaseModel):
    file_name: str
    saved: bool

class ProcessRequest(BaseModel):
    saved: bool
    files: List[str]
    prompt: str
    positive: bool
    highlight: bool

class DeleteFile(BaseModel):
    file_name: str
    file_status: str

class ShareFileRequest(BaseModel):
    file: str
    file_recipient: str
