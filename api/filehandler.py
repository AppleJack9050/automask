import os
import shutil
import zipfile
import tarfile
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
from filesTable import FilesTable
import io

#TODO fix file names

class FileHandler:
    def __init__(self, directory, processed_directory):
        self.upload_directory = directory
        self.processed_directory = processed_directory
        self.saved_directory = "/saved"
        self.file_table = FilesTable()

    def handle_zip(self, user, zip_file, file_name):
        file_name = self.file_table.insert_name(file_name, user)
        temp_directory = os.path.join(self.upload_directory, "_temp_extract")
        zip_path = os.path.join(temp_directory, file_name)

        with open(zip_path, "wb") as f:
            f.write(zip_file.file.read())

        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall(temp_directory)

        os.remove(zip_path)
        self.__extract_temp_directory(user, temp_directory)

    def handle_tar(self, user, tar_file, file_name):
        temp_directory = os.path.join(self.upload_directory, "_temp_extract")
        os.makedirs(temp_directory, exist_ok=True)
        file_name = self.file_table.insert_name(file_name, user)
        tar_path = os.path.join(temp_directory, file_name)

        with open(tar_path, "wb") as f:
            f.write(tar_file.file.read())

        with tarfile.open(tar_path, "r:*") as tar_ref:
            tar_ref.extractall(temp_directory)

        os.remove(tar_path)
        self.__extract_temp_directory(user, temp_directory)

    def __extract_temp_directory(self, user, temp_directory):
        for root, _, files in os.walk(temp_directory):
            for file in files:
                if file.startswith("._"):
                    continue

                src_path = os.path.join(root, file)

                new_name = self.file_table.insert_name(file_name=file, user=user)
                base, ext = os.path.splitext(new_name)
                dest_path = os.path.join(self.upload_directory, user, new_name)

                count = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(self.upload_directory, user, f"{base}_{count}{ext}")
                    count += 1

                shutil.move(src_path, dest_path)

        shutil.rmtree(temp_directory)

    async def handle_single_file(self, user, file, file_name):
        file_name = self.file_table.insert_name(file_name, user)
        file_path = os.path.join(self.upload_directory, user, file_name)
        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

    async def handle_multiple_files(self, user, files):
        saved_files = []
        for file in files:
            filename = file.filename
            match filename:
                case file_name if filename.endswith(".zip"):
                    self.handle_zip(user, file, filename)
                case file_name if filename.endswith(".tar.gz") or filename.endswith(".tar") or filename.endswith(".tgz"):
                    self.handle_tar(user, file, filename)
                case _:
                    await self.handle_single_file(user, file, filename)

        return saved_files

    def fetch_image(self, user, image_title, edited_image = False):
        stored_file_name = self.file_table.get_stored_name(image_title, user)
        stored_file_name = Path(stored_file_name).stem

        file_path = os.path.join(self.processed_directory, user, Path(stored_file_name).stem)
        if edited_image:
            file_path = os.path.join(file_path, "edited")
        else:
            file_path = os.path.join(file_path, "original")

        for root, _, files in os.walk(file_path):
            for file in files:
                if stored_file_name in file:
                    image = Image.open(os.path.join(root, file))
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    return  base64.b64encode(buffered.getvalue()).decode("utf-8")
        return None

    def return_image_masks(self, user, file_name):
        stored_file_name = self.file_table.get_stored_name(file_name, user)
        file_path = os.path.join(self.processed_directory, user, Path(stored_file_name).stem)

        masks = []

        for root, _, files in os.walk(Path(file_path) / "masks"):
            for index, file in enumerate(files):
                    with open(os.path.join(root, file), "rb") as mask:
                        masks.append(
                        {
                            "mask":base64.b64encode(mask.read()).decode("utf-8"),
                            "selected":False,
                            "opacity":0,
                            "name":index
                        })
                        
        return masks

    def save(self, user, file, file_name, file_type):
        stored_file_name = self.file_table.get_stored_name(file_name, user)
        if not os.path.exists(os.path.join(self.saved_directory, user)):
            os.makedirs(os.path.join(self.saved_directory, user))

        image_data = base64.b64decode(file)

        file_path = Path(self.saved_directory) / user / f"{stored_file_name}"
        with open(file_path.resolve(), "wb") as image:
            image.write(image_data)

    def download(self, user, file_name) -> str:
        file_path = os.path.join(self.saved_directory, user, self.file_table.get_stored_name(file_name))
        image = Image.open(file_path)
        image.thumbnail((image.size))
        buffered = BytesIO()
        
        image.save(buffered, format=(image.format or "PNG"))
        
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'

    def list_files(self, user, producer) -> list:
        os.makedirs(os.path.join(self.upload_directory, user), exist_ok=True)
        os.makedirs(os.path.join(self.processed_directory, user), exist_ok=True)
        os.makedirs(os.path.join(self.saved_directory, user), exist_ok=True)

        unprocessed_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, user),
                os.listdir(os.path.join(self.upload_directory, user))
        ))
        processed_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, user),
                os.listdir(os.path.join(self.processed_directory, user))
        ))

        saved_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, user),
                os.listdir(os.path.join(self.saved_directory, user))
        ))
        files_being_processed = producer.check_queue(user)

        for file in files_being_processed:
            if file in unprocessed_files:
                unprocessed_files.remove(file)

        return [
            {"Unprocessed": unprocessed_files},
            {"Processing": files_being_processed},
            {"Processed": processed_files},
            {"Saved": saved_files}
        ]

    def delete_file(self, user, file_name):
        stored_file_name = self.file_table.get_stored_name(file_name)
        stored_file_stem = Path(stored_file_name).stem
    
        unproccessed_file_path = os.path.join(self.upload_directory, user, stored_file_name)
        proccessed_file_path = os.path.join(self.processed_directory, user, stored_file_stem)
        saved_file_path = os.path.join(self.saved_directory, user, stored_file_name)

        if os.path.exists(unproccessed_file_path):
            os.remove(unproccessed_file_path)

        if os.path.exists(proccessed_file_path):
            os.remove(proccessed_file_path)

        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        return

    def download_zip_file(self, user, files):
        return self.__create_zip_file(self.__get_files_and_stored_path(user, files))

    def download_tar_file(self, user, files):
        return self.__create_tar_file(self.__get_files_and_stored_path(user, files))

    def __create_zip_file(self, files):
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip:
            for file in files:
                zip.write(file["file_path"], arcname=file["actual_name"])
        buffer.seek(0)
        yield buffer.getvalue()

    def __create_tar_file(self, files):
        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            for file in files:
                tar.add(file["file_path"], arcname=file["actual_name"])
                
        buffer.seek(0)
        yield buffer.getvalue()

    def __get_files_and_stored_path(self, user, files):
        return list(
            map(
                lambda file: 
                {
                    "file_path":os.path.join(
                        self.saved_directory,
                        user,
                        self.file_table.get_stored_name(file, user)
                    ),
                    "actual_name":file 
                },
                files
        ))
