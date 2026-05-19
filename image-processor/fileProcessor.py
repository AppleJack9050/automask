import queue
import shutil
import os
from pathlib import Path
import torch
from datetime import datetime
from fileEditor import FileEditor
from sam2Processor import SAM2Processor
from sam3Processor import SAM3Processor

class FileProcessor():
    def __init__(self, upload_directory, processed_directory, saved_directory):
        self.upload_directory = upload_directory
        self.processed_directory = processed_directory
        self.saved_directory = saved_directory
        self.file_q = queue.Queue()
        self.file_editor = FileEditor(processed_directory)
        self.device = self.__get_device_for_SAM()
        os.makedirs(processed_directory, exist_ok=True)
        self.inference_model = self.__select_sam_model()

    def __select_sam_model(self) -> SAM3Processor | SAM2Processor:
        if os.getenv("USE_SAM3") == "1":
            print("Using SAM3...")
            return SAM3Processor(self.device, self.upload_directory)
        print("Using SAM2...")
        return SAM2Processor(self.device, self.upload_directory)

    def __get_device_for_SAM(self) -> str:
        device = ""
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def create_process_queue(self, user: str, files: str):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')) and file in files:
                self.file_q.put(file)
                self.__make_processed_file_folder(user, file)

    def process_files_in_queue(self,  user: str, prompt: str, positive: bool, highlight: bool, saved: bool):
        print(f'Started Processing {datetime.now()}')
        original_upload_dir = self.upload_directory

        if saved:
            self.inference_model.upload_directory = self.saved_directory

        while not self.file_q.empty():
            try:
                self.__process_file(user, self.file_q.get(), prompt, positive, highlight, saved)
            except Exception:
                pass
            finally:
                self.file_q.task_done()
    
        print(f'Finished Processing {datetime.now()}')
        self.inference_model.upload_directory = original_upload_dir

    def __process_file(self, user: str, file: str, prompt: str, positive: bool, highlight: bool, saved: bool):
        self.target_processed_folder = Path(self.processed_directory) / user / Path(file).stem
        mask_dir = self.target_processed_folder / "masks"
        mask_dir = mask_dir.resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)

        original_copy_dir = self.target_processed_folder / "original"
        original_copy_dir = original_copy_dir.resolve()
        original_copy_dir.mkdir(parents=True, exist_ok=True)

        if prompt == '':
            self.inference_model.process_empty_prompt(user, mask_dir, file)
            self.__move_processed_file(user, file, saved)

        else:
            self.inference_model.process_text_prompt(user, mask_dir, file, prompt)
            self.__move_processed_file(user, file, saved)

            self.file_editor.edit_image(file, self.target_processed_folder, positive, highlight)

    def __move_processed_file(self, user: str, file: str, saved: bool):
        if not saved:
            shutil.move(
                str(Path(self.upload_directory) / user / file),
                str(self.target_processed_folder / "original" / f"{file}")
            )

    def __make_processed_file_folder(self, user: str, file: str):
        processed_dir = Path(self.processed_directory) / user / Path(file).stem
        processed_dir.mkdir(parents=True, exist_ok=True)
