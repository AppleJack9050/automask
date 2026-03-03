import os
import shutil
import zipfile
import tarfile
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO

class FileHandler:
    def __init__(self, directory, processed_directory):
        self.directory = directory
        self.processed_directory = processed_directory
        self.saved_directory = "./saved"

    def handle_zip(self, zip_file, filename):
        temp_directory = os.path.join(self.directory, "_temp_extract")
        os.makedirs(temp_directory, exist_ok=True)

        zip_path = os.path.join(temp_directory, filename)

        with open(zip_path, "wb") as f:
            f.write(zip_file.file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_directory)

        for root, _, files in os.walk(temp_directory):
            for f in files:
                if f.startswith('._') or f.endswith('.zip'):
                    continue
                src_path = os.path.join(root, f)
                dest_path = os.path.join(self.directory, f)
                base, ext = os.path.splitext(f)
                count = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(self.directory, f"{base}_{count}{ext}")
                    count += 1
                shutil.move(src_path, dest_path)
        shutil.rmtree(temp_directory)

    def handle_tar(self, tar_file, filename):
        temp_directory = os.path.join(self.directory, "_temp_extract")
        os.makedirs(temp_directory, exist_ok=True)

        tar_path = os.path.join(temp_directory, filename)

        with open(tar_path, "wb") as f:
            f.write(tar_file.file.read())

        with tarfile.TarFile(tar_path, "r") as tar_ref:
            tar_ref.extractall(temp_directory)

        for root, _, files in os.walk(temp_directory):
            for f in files:
                src_path = os.path.join(root, f)
                dest_path = os.path.join(self.directory, f)
                base, ext = os.path.splitext(f)
                count = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(self.directory, f"{base}_{count}{ext}")
                    count += 1
                shutil.move(src_path, dest_path)
        shutil.rmtree(temp_directory)

    async def handle_single_file(self, file, filename):
        file_path = os.path.join(self.directory, filename)
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

    def fetch_image(self, path, image_title):
        for root, _, files in os.walk(path):
            for file in files:
                if image_title in file:
                    image = Image.open(os.path.join(root, file))
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    return  base64.b64encode(buffered.getvalue()).decode("utf-8")
        return None

    def return_image_masks(self, path):
        masks = []

        for root, _, files in os.walk(Path(path) / "masks"):
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
        if not os.path.exists(self.saved_directory):
            os.makedirs(self.saved_directory)

        image_data = base64.b64decode(file)

        file_path = Path(self.saved_directory) / f"{file_name}{file_type}"
        with open(file_path.resolve(), "wb") as image:
            image.write(image_data)

    def download(self, file_path):
        _, file_type = os.path.splitext(file_path)
        image = Image.open(file_path)
        image.thumbnail((image.size))
        buffered = BytesIO()
        
        image.save(buffered, format=(image.format or "PNG"))
        
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'
