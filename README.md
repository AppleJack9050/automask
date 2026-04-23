# AutoMask Application Quick Start Guide
## If building for the first time 
To build the images from scratch you will have to make sure that you have Grounded-SAM-2[https://github.com/IDEA-Research/Grounded-SAM-2/tree/main] downloaded, and inside of the image-processor subfolder as this is where the image stores the model. You don't need to run any of the scripts to install it this is all handled in the Dockerfiles. Building may take a little while, as the models and all their dependenices are quite large.

**If you are using a mac M1 or later images are supplied here[]**

## Running With Docker Compose (Recommended)

1. Make sure docker and groundedSAM2 are installed correctly.

2. run  ```docker compose up```

## Running With Kubernetes
Otherwise if you have the extra power, to run with Kubernetes:
1. Install Minikube[https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download] for local cluster management.

2. Install Helm[https://helm.sh/docs/intro/install/] (Used for KEDA)

3. run ```minukube start```

4. Install KEDA[https://keda.sh/docs/2.19/deploy/]

5. run ```kubectl apply -f k8s```

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
