# AutoMask Application Quick Start Guide

All you need to do normally is to run  ```docker compose up```.

How if you are looking to build the images from scratch you will have to make sure that you have [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2/tree/main) downloaded, and inside of the api subfolder as this is where the image stores the model. You don't need to run any of the scripts to install it this is all handled in the Dockerfiles.

## Usage Guide

### Step 1: Upload Files

1. Navigate to the **Input** section of the application.

2. **Drag and drop** one or more image files into the designated upload area.

### Processing Files

1. Go to the **View Files** section.

2. Click the **Process Files** button.

3. This step prepares the  images for editing, generating the masks. Wait a bit until the files move from the 'Input' list to the **Processed** (would need to refresh the page)

### Editing

1. From the **Processed** file list, click the **Edit** button next to the file you wish to modify.

2. This will open the **Edit File** view where you can perform detailed object segmentation and modification.

3. The touchup tool, available through the context menu allows for additional manual removal. through a brush like UI, left click and hold to remove pixels from the image and is adjustable through a scale under the image. To exit this right click and normal editing will resume.

### Editing Actions (Right-Click Menu)

In the editor, right click on any segmented object to:

* **Remove from Image:** Select this to erase the object and remove it from the image.

* **Select Only the Object:** Select this to isolate and highlight only the selected object.

### Downloading

1. **right click** anywhere on the image and select **Save**.

2. A modal will appear, prompting you to choose your desired output file type.

3. Select the format and save the processed image.
