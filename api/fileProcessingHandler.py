import os
import queue
import numpy as np
import shutil
import os
import numpy as np
from pathlib import Path
import torch
from datetime import datetime
import logging
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from dotenv import load_dotenv
import gc

logger = logging.getLogger(__name__)

class FileProcessor():
    def __init__(self, target_directory, processed_directory):
        self.target_directory = target_directory
        self.processed_directory = processed_directory
        self.file_q = queue.Queue()
        self.device = self.__get_device_for_SAM()

#        self.device="cpu" 
        logging.basicConfig(filename='fileprocessor.log', level=logging.INFO)
        os.makedirs(processed_directory, exist_ok=True)
        load_dotenv() 

        self.sam2_model = self.__build_sam2()

    def __get_device_for_SAM(self):
        device = ""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def file_processed(self, file) -> bool:
        path_to_check = os.path.join(self.processed_directory, file)
        return not os.path.exists(path_to_check)

    def create_process_queue(self):
        files = os.listdir(self.target_directory)
        unprocessed_files = filter(self.file_processed, files)
        for file in unprocessed_files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.file_q.put(file)

    def process_files_in_queue(self):
        logger.info(f'Started Processing {datetime.now()}')
        while not self.file_q.empty():
            self.__process_file(self.file_q.get())

    def __process_file(self, file):
        logger.info(f'Processing {file}')      
        processed_dir = Path(self.processed_directory)
        target_processed_folder = processed_dir / Path(file).stem

        target_processed_folder.mkdir(parents=True, exist_ok=True)
        mask_dir = target_processed_folder / "masks"
        mask_dir = mask_dir.resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)        
        masks = self.__generate_sam2_masking(str(Path(self.target_directory) / file))
        os.remove(str(Path(self.target_directory) / file))

        masks["image"].save(target_processed_folder / f"{file}.jpg")
        for index, mask in enumerate(masks["masks"]):
            mask_image = mask["segmentation"]

            if mask_image.dtype == bool:
                mask_image = mask_image.astype(np.uint8) * 255

            mask_image = Image.fromarray(mask["segmentation"])
            mask_image.save(mask_dir / f"mask_{index}.png")
            logger.info(f"Saving Mask {index} at: {mask_dir / f"mask_{index}.png"}")

        del masks
        gc.collect()

    def __generate_sam2_masking(self, image):
        """Generates the object masks from sam2"""
        image = Image.open(image)
        image = image.resize((1024, 1024))
        #image = image.resize((512, 512))
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
        masks = mask_generator.generate(image_np)
        del mask_generator
        gc.collect()
        return{"image": image, "masks": masks}

    def __build_sam2(self):
        cwd = os.getcwd()
        os.chdir(os.getenv("SAM2_FULL_PATH"))
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        os.chdir(cwd)
        return sam2_model


