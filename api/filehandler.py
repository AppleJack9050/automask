import os
import shutil
import zipfile
import tarfile
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
from filesTable import FilesTable

class FileHandler:
    def __init__(self, directory, processed_directory):
        self.directory = directory
        self.processed_directory = processed_directory
        self.saved_directory = "/saved"
        self.file_table = FilesTable()

    def handle_zip(self, zip_file, file_name):
        temp_directory = os.path.join(self.directory, "_temp_extract")
        os.makedirs(temp_directory, exist_ok=True)

        zip_path = os.path.join(temp_directory, file_name)

        with open(zip_path, "wb") as f:
            f.write(zip_file.file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_directory)

        self.__extract_temp_directory(temp_directory)

    def handle_tar(self, tar_file, file_name):
        temp_directory = os.path.join(self.directory, "_temp_extract")
        os.makedirs(temp_directory, exist_ok=True)
        file_name = self.file_table.insert_name(file_name, "admin")

        tar_path = os.path.join(temp_directory, file_name)

        with open(tar_path, "wb") as f:
            f.write(tar_file.file.read())

        with tarfile.TarFile(tar_path, "r") as tar_ref:
            tar_ref.extractall(temp_directory)

        self.__extract_temp_directory(temp_directory)

    def __extract_temp_directory(self, temp_directory):
        for root, _, files in os.walk(temp_directory):
            for f in files:
                f = self.file_table.insert_name(f, "admin")
                src_path = os.path.join(root, f)
                dest_path = os.path.join(self.directory, f)
                base, ext = os.path.splitext(f)
                count = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(self.directory, f"{base}_{count}{ext}")
                    count += 1
                shutil.move(src_path, dest_path)
        shutil.rmtree(temp_directory)

    async def handle_single_file(self, file, file_name):
        file_name = self.file_table.insert_name(file_name, "admin")
        file_path = os.path.join(self.directory, file_name)
        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

    async def handle_multiple_files(self, files):
        saved_files = []
        for file in files:
            filename = file.filename
            match filename:
                case file_name if filename.endswith(".zip"):
                    self.handle_zip(file, filename)
                case file_name if filename.endswith(".tar.gz"):
                    self.handle_tar(file, filename)
                case _:
                    await self.handle_single_file(file, filename)

        return saved_files

    def fetch_image(self, image_title, edited_image = False):
        stored_file_name = self.file_table.get_stored_name(image_title, "admin")
        stored_file_name = Path(stored_file_name).stem

        file_path = os.path.join(self.processed_directory, Path(stored_file_name).stem)
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

    def return_image_masks(self, file_name):
        stored_file_name = self.file_table.get_stored_name(file_name, "admin")
        file_path = os.path.join(self.processed_directory, Path(stored_file_name).stem)

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

    def save(self, file, file_name, file_type):
        stored_file_name = self.file_table.get_stored_name(file_name)
        if not os.path.exists(self.saved_directory):
            os.makedirs(self.saved_directory)

        image_data = base64.b64decode(file)

        file_path = Path(self.saved_directory) / f"{stored_file_name}{file_type}"
        with open(file_path.resolve(), "wb") as image:
            image.write(image_data)

    def download(self, file_name):
        file_path = os.path.join(self.saved_directory, self.file_table.get_stored_name(file_name))

        _, file_type = os.path.splitext(file_path)
        image = Image.open(file_path)
        image.thumbnail((image.size))
        buffered = BytesIO()
        
        image.save(buffered, format=(image.format or "PNG"))
        
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'

    def list_files(self, producer) -> list:
        unprocessed_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, "admin"),
                os.listdir(self.directory)
        ))
        processed_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, "admin"),
                os.listdir(self.processed_directory)
        ))
        saved_files = list(
            map(
                lambda stored_name: self.file_table.get_actual_name(stored_name, "admin"),
                os.listdir(self.saved_directory)
        ))
        files_being_processed = producer.check_queue()

        for file in files_being_processed:
            if file in unprocessed_files:
                unprocessed_files.remove(file)

        return [
            {"Unprocessed": unprocessed_files},
            {"Processing": files_being_processed},
            {"Processed": processed_files},
            {"Saved": saved_files}
        ]

    def delete_file(self, file_path):
        pass
