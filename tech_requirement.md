# Interactive Image Background Removal System (Based on SAM2)
### Software Requirements Specification (v1.5)

---

## 1. Project Overview
This project aims to develop an **interactive research-oriented image background removal system** based on **Segment Anything Model 2 (SAM2)**.  
The system is designed for researchers and computer vision practitioners to extract image foregrounds and remove backgrounds for both **single images** and **entire datasets**.  
It supports **drag-and-drop dataset upload**, **GPU-accelerated inference**, **interactive masking and editing**, **undo/redo and full restoration**, and **multi-format export**.  
The architecture follows a **frontend-backend separation**, ensuring high performance, extensibility, and scientific reproducibility.

---

## 2. Product Positioning
- **Product Type**: Research-grade image segmentation and background removal tool.
- **Core Objectives**:
  - Provide high-precision background removal for single images and large-scale datasets.
  - Support real-time interactive masking, undo/redo, and restoration.
  - Enable drag-and-drop uploads and bulk downloads.
  - Ensure stability, reproducibility, and usability in scientific workflows.

---

## 3. Functional Requirements

### 3.1 Image and Dataset Upload
- Users can import data via **file selection** or **drag-and-drop**, supporting:
  - Single image files;
  - Entire folders (datasets);
  - Compressed archives (.zip / .tar.gz).
- The drag-and-drop area must be clearly visible and support multiple files or directories.
- Supported formats: **JPG, PNG, WEBP, TIFF**.
- After upload, the system parses the directory structure and builds a task list.
- For single images: users directly enter the interactive editing mode.
- For datasets: a dataset management view is displayed where users select how to process images.
- **Parallel file upload** is supported (multiple simultaneous transfers), with progress visualization.

---

### 3.2 Batch Processing
- After dataset upload, each image is added to a task list.
- Users manually start processing tasks.
- The system supports **parallel uploads** but **sequential processing** only, to ensure GPU stability and consistent results.
- Image status is **manually labeled by the user**, not auto-detected by the system:
  - Each image can be marked as **“To-Do / In-Progress / Completed”**.
  - These labels are for manual workflow tracking and do not affect inference execution.
- During processing, users may switch to previously handled images to inspect or refine results.
- Once all tasks are complete, users may:
  - Download any individual image result;
  - Or download the **entire dataset as a single ZIP/TAR.GZ package**, preserving original folder hierarchy.

---

### 3.3 Real-Time Segmentation and Interactive Editing
- As the user moves the mouse, the system displays **real-time segmentation overlays** (semi-transparent foreground previews).
- Right-click context menu operations:
  - **Mask** – mark the region for background removal.
  - **Unmask** – remove the region from masking (optional).
- Changes apply immediately, with live canvas updates.
- Each image maintains its own independent editing session and save state.

---

### 3.4 Undo, Redo, and Restore Original Image
- The system supports **undo** and **redo** operations.
- Users can revert to previous states at any point.
- **History depth: up to 10 steps.**
- A **“Restore Original”** function resets the image to its initially uploaded version, clearing all masks and edit history.
- Each image has an independent undo stack and original image cache.
- Edit history can be exported or replayed for scientific reproducibility.

---

### 3.5 Result Export
- Users can export results for single images or entire datasets.
- Supported export formats: **PNG, JPG, BMP, TIFF**.
- Export modes:
  - **Transparent Background** – retains only the foreground (ideal for PNG/TIFF).
  - **Black Mask Output** – binary mask, white foreground and black background.
  - **Overlay Preview** – shows masked regions overlaid in color.
- When exporting a dataset, the directory structure of the results matches the original dataset.

---

## 4. User Interface and Interaction

### 4.1 Main Interface
- Main components:
  - Drag-and-drop upload area (accepts files and folders);
  - Upload progress and task list;
  - Single-image editing canvas;
  - Batch export and download controls.
- Mouse hover shows segmentation outlines in real time.
- Users can manually label each image as **To-Do / In-Progress / Completed**.
- Right-click menu includes:
  - Mask
  - Unmask (optional)
  - Cancel
- Top toolbar includes:
  - Undo, Redo
  - Restore Original
  - Export
  - Toggle Mask Visibility
  - View Original

### 4.2 Visual Feedback
- Real-time segmentation overlay: semi-transparent colored mask (30–50% opacity).
- Masked regions displayed as black or transparent.
- Users can freely navigate to any “Completed” image to verify or adjust results.

---

## 5. Non-Functional Requirements

| Category | Specification |
|-----------|----------------|
| **Performance** | Supports parallel file uploads; interactive latency ≤ 200 ms per image. |
| **Accuracy** | Segmentation output must be identical to the raw SAM2 model output, without interpolation loss. |
| **Stability** | Capable of handling large datasets (≥ 1,000 images) sequentially without GPU overflow. |
| **Usability** | Drag-and-drop upload, masking, and restoration must be intuitive. |
| **Compatibility** | Works on Chrome, Edge, and Firefox. |
| **Security** | File type and size validation; prevent non-image or script uploads. |
| **Extensibility** | Modular model interface supporting replacement of segmentation back-ends. |

---

## 6. System Architecture Requirements
- **Frontend**: Handles user interaction, drag-and-drop upload, task labeling, and rendering.
- **Backend**: Manages file reception, task control, and SAM2 inference.
- **Model Service**: GPU-based container executing tasks sequentially with task queue buffering.
- **Storage Layer**: Local filesystem or object storage (e.g., MinIO / S3).
- **Deployment**: Docker-based container deployment compatible with research compute environments.

---

## 7. Operation Workflow
1. User drags single images or datasets into the upload area.
2. The system analyzes the structure and generates a task list.
3. The user manually labels task statuses (To-Do / In-Progress / Completed).
4. The user starts inference or enters interactive editing for any image.
5. The user may switch to completed images for verification or fine-tuning.
6. During editing, masking, undo/redo, and **restore original** actions are available.
7. The user exports single results or the entire dataset.

---

## 8. System Constraints
- Parallel upload supported; image processing is sequential.
- Image status labeling is manual only.
- GPU environment required for model inference.
- Video files are not supported.

---

## 9. Future Expansion (Optional)
- **Multi-User Task and Permission Management**: enable multiple researchers to maintain separate projects, histories, and access permissions.

---

## 10. Acceptance Criteria
- [x] Supports drag-and-drop upload for single images and datasets.  
- [x] Allows users to manually label image statuses.  
- [x] Supports parallel uploads and sequential processing.  
- [x] Users can navigate and inspect previously completed images.  
- [x] Real-time segmentation previews display correctly.  
- [x] Supports masking, undo/redo, and restore-to-original actions (10-step history).  
- [x] Exports in PNG, JPG, BMP, and TIFF formats.  
- [x] Dataset exports maintain original directory structure.  
- [x] System runs stably and meets research-grade precision and reproducibility.

---
