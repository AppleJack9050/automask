# AutoMask Application Quick Start Guide

This guide provides instructions for setting up and running the AutoMask API (backend) and Vue web application (frontend).

## Prerequisites and Environment Setup

### 1. Conda Environment Management

We recommend using **Conda** for isolated and efficient package management.

* **Conda Installation:** If you do not have Conda installed, please follow the official installation guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

* **Create Environment:** Use the following command to create a new Conda environment named `automask-env` and install all required Python dependencies from `requirements.txt`:

```conda create --name automask-env --file requirements.txt```

### 2. Frontend Dependencies (npm)

The frontend requires **Node Package Manager**.

* **npm Installation:** Ensure you have npm installed by following the guide [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

## API and Webapp Quick Start

### 1. Starting the Backend API

```cd ./api```
```uvicorn endpoints:app --reload```

### 2. Starting the Frontend Webapp (Vue)

cd ./automask-frontend

**Install** the necessary Node modules:

```npm install```

**Start** the development server:

```npm run dev```

The web application will start at`http://localhost:5173`

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
