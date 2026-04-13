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
from fileEditor import FileEditor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict, box_convert

import gc

logger = logging.getLogger(__name__)

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

class FileProcessor():
    def __init__(self, upload_directory, processed_directory, saved_directory):
        self.upload_directory = upload_directory
        self.processed_directory = processed_directory
        self.saved_directory = saved_directory
        self.file_q = queue.Queue()
        logging.basicConfig(filename='fileprocessor.log', level=logging.INFO)
        self.file_editor = FileEditor(processed_directory)
        self.device = self.__get_device_for_SAM()

        os.makedirs(processed_directory, exist_ok=True)

        self.sam2_model = self.__build_sam2()
        self.dino = self.__build_dino()

    def __get_device_for_SAM(self) -> str:
        device = ""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
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
            self.upload_directory = self.saved_directory

        while not self.file_q.empty():
            try:
                self.__process_file(user, self.file_q.get(), prompt, positive, highlight, saved)
            except Exception:
                pass
            finally:
                self.file_q.task_done()
    
        print(f'Finished Processing {datetime.now()}')
        self.upload_directory = original_upload_dir

    def __process_file(self, user: str, file: str, prompt: str, positive: bool, highlight: bool, saved: bool):
        logger.info(f'Processing {file}')
        self.target_processed_folder = Path(self.processed_directory) / user / Path(file).stem
        mask_dir = self.target_processed_folder / "masks"
        mask_dir = mask_dir.resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)

        original_copy_dir = self.target_processed_folder / "original"
        original_copy_dir = original_copy_dir.resolve()
        original_copy_dir.mkdir(parents=True, exist_ok=True)

        if prompt == '':
            logger.info(f'Using sam2 only {datetime.now()}')
            self.__process_sam2(user, mask_dir, file)
            self.__move_processed_file(user, file, saved)

        else:
            logger.info(f'Using grounded-sam2 {datetime.now()}')
            self.__process_grounded_sam2(user, mask_dir, file, prompt)
            self.__move_processed_file(user, file, saved)

            logger.info(f'Editing {file}')
            self.file_editor.edit_image(file, self.target_processed_folder, positive, highlight)

    def __process_grounded_sam2(self, user: str, mask_dir: str, file: str, prompt: str):
        masks = self.__generate_grounded_sam2_mask(prompt, str(Path(self.upload_directory) / user / file))

        for index, mask in enumerate(masks):
            mask_image = np.asarray(mask)
            if mask.dtype != np.uint8:
                mask = np.clip(mask, 0, 1) * 255
                mask = mask.astype(np.uint8)

            if mask.ndim == 3:
                mask = mask.squeeze()

            mask_image = Image.fromarray(mask, mode="L")
            mask_image.save(mask_dir / f"mask_{index}.png")

            logger.info(f"Saving Mask {index} at: {mask_dir} / mask_{index}.png")

        del masks
        gc.collect()

    def __process_sam2(self, user: str, mask_dir: str, file: str):
        masks = self.__generate_sam2_masking(str(Path(self.upload_directory) / user / file))

        for index, mask in enumerate(masks):
            mask_image = mask["segmentation"]

            if mask_image.dtype == bool:
                mask_image = mask_image.astype(np.uint8) * 255

            mask_image = Image.fromarray(mask["segmentation"])
            mask_image.save(mask_dir / f"mask_{index}.png")
            logger.info(f"Saving Mask {index} at: {mask_dir} / mask_{index}.png")
        
        del masks
        gc.collect()

    def __generate_sam2_masking(self, image: str):
        """Generates the object masks from sam2"""
        image = Image.open(image)

        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
        masks = mask_generator.generate(image_np)
        del mask_generator
        gc.collect()
        
        return masks

    def __build_sam2(self):
        cwd = os.getcwd()
        os.chdir(os.getenv("SAM2_FULL_PATH"))
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        os.chdir(cwd)
        return sam2_model

    def __build_dino(self):
        return load_model(
            model_config_path=os.getenv("GROUNDING_DINO_CONFIG"),
            model_checkpoint_path=os.getenv("GROUNDING_DINO_CHECKPOINT"),
            device=self.device
        )

    def __generate_grounded_sam2_mask(self, prompt: str, image: str):
        sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        image_source, image = load_image(image)
        sam2_predictor.set_image(image_source)
        prompt = prompt.lower()
        if not prompt.endswith('.'):
            prompt = prompt + '.'

        box_prompts, _, labels = predict(
            model=self.dino,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.device
        )

        h, w, _ = h, w, _ = image_source.shape
        box_prompts = box_prompts* torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=box_prompts, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks = []
        for box in input_boxes:
            box_masks, scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks.append(box_masks)

        return masks

    def __move_processed_file(self, user: str, file: str, saved: bool):
        if not saved:
            shutil.move(
                str(Path(self.upload_directory) / user / file),
                str(self.target_processed_folder / "original" / f"{file}")
            )

    def __make_processed_file_folder(self, user: str, file: str):
        processed_dir = Path(self.processed_directory) / user / Path(file).stem
        processed_dir.mkdir(parents=True, exist_ok=True)
