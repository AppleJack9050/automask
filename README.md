# AutoMask Application Quick Start Guide

First to start make sure that you have sam2 downloaded, and inside of the api subfolder as this is where docker pulls it from.

All you need to to do run it next is use ```docker compose up```. Because of SAM2 this image is a big boy so it may take a little while to build
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

### Editing Actions (Right-Click Menu)

In the editor, right click on any segmented object to:

* **Remove from Image:** Select this to erase the object and remove it from the image.

* **Select Only the Object:** Select this to isolate and highlight only the selected object.

### Downloading

1. **right click** anywhere on the image and select **Save**.

2. A modal will appear, prompting you to choose your desired output file type.

3. Select the format and save the processed image.
