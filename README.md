# AutoMask Application Quick Start Guide
## If building for the first time 
### Using SAM3
1. To install SAM3 the instructions are provided here[https://github.com/facebookresearch/sam3/tree/main]. This should be cloned into the image processor subfolder.
2. To download the checkpoints access on huggingface is required to the repo[https://huggingface.co/facebook/sam3.1]. 

3. Create a hf access token.

4. Create a .env file in the image processor subfolder, and set the HF_TOKEN environment variable to be the access token made


**Note** 
If using SAM3 without a CUDA environment, some modifications to the SAM3 source code is required.

### Using SAM2
To build the images from scratch you will have to make sure that you have Grounded-SAM-2[https://github.com/IDEA-Research/Grounded-SAM-2/tree/main] downloaded, and inside of the image-processor subfolder as this is where the image stores the model. You don't need to run any of the scripts to install it this is all handled in the Dockerfiles. Building may take a little while, as the models and all their dependenices are quite large.
### DockerCompose File
Set the ```USE_SAM3``` environment variable to 0.

## Running With Docker Compose (Recommended)

1. Make sure docker and either SAM3 or groundedSAM2 are installed correctly.

2. run  ```docker compose up```

## Running With Kubernetes
These images are built for arm (Mac) architecture, it may require building, tagging and using docker hub if using on alternative formats.

Otherwise if you have the extra power, to run with Kubernetes:
1. If Using SAM3, set the ```HF_TOKEN``` in the congigmap.yml to be your token and set ```USE_SAM3``` to '1'.

2. Install Minikube[https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download] for local cluster management.

3. Install Helm[https://helm.sh/docs/intro/install/] (Used for KEDA)

4. run ```minukube start```

5. Install KEDA[https://keda.sh/docs/2.19/deploy/]

6. run ```kubectl apply -f k8s```

## Usage Guide
### Datasets
#### Single Image Evaluation
SAM2 Does work on any images and any prompts, although here if you are looking to recreate my results here is the full dataset[]. Although as each image is huge it will be very time consuming to run and wait for the full dataset. Instead use these detailed images for looking at prompt performance on a few sample images of your choosing.

#### For Full Dataset Comparison
Here is a compressed version of the same datset in which if looking to use this dataset along with COLMAP[] will achieve must faster inference.
### Step 1: Upload Files

1. Navigate to the **Input** section of the application.

2. **Drag and drop** one or more image or archive files into the designated upload area.

### Processing Files

1. Go to the **View Files** page.

2. Look at the 'Uploads' section.

3. Click either select all or select each file you wish to process.

4. Click the **Process Files** button.

5. Upon the Modal, enter your prompt if you wish to use one or leave it empty for full segmentation.

6. Select the given editing action you wish to carry out before clicking **process**.

7. This step prepares the  images for editing, generating the masks. Files will move in groups of 50 from **uploads** to **processing**, and then to **processed**. To see this happen use the "Refresh Files" button located at the top of the page.

### Editing

1. From the **Processed** file list, click the **Edit** button next to the file you wish to modify.

2. This will open the **Edit File** view where you can perform detailed object segmentation and modification.

3. The touchup tool, available through the context menu allows for additional manual removal. through a brush like UI, left click and hold to remove pixels from the image and is adjustable through a scale under the image. To exit this right click and normal editing will resume.

4. To save the current state of the object click the **Save** button and select the file type.

### Downloading

1. In the **File View** Page navigate to the Saved section. 

2. Select the files you wish to download.

3. Select the archive file name. (This will have no effect when only one file is selected as it only downloads that).
