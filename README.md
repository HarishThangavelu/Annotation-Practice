Nuclio Auto-Annotation Pipeline – Clean Professional Version
1. Train the Model and Generate Weights

Before setting up the Nuclio function, the object detection model must be trained.
This can be done using:

YOLOv8 / YOLOv5

MMRotate (for orientation-aware detection)

Any compatible PyTorch model

Output:
✔ A trained weight file (best.pt or similar)
✔ A class list used during training

These weights will be mounted inside the Nuclio container for inference.

2. Create the Nuclio Inference Script (main.py)

This script handles:

Loading the trained model

Preprocessing incoming images

Running inference

Formatting predictions in CVAT-compatible JSON

Example template logic:

# placeholder example
# load model
# preprocess image
# run inference
# convert predictions to CVAT format
# return JSON response


The real inference logic remains confidential and is not included.

3. Create the Nuclio Function Specification (function.yaml)

The YAML file defines:

The runtime (Python)

Required Python libraries (e.g., ultralytics, torch)

Class names required by the model

Volumes for mounting model weights

Handler name

Resource limits (CPU, RAM)

This file is used by Nuclio to configure how the model runs inside a container.

Installation Options

Nuclio dependencies can be installed in two ways:

Option A — Using requirements.txt in the YAML

Nuclio installs Python packages automatically.

Option B — Installing inside the Docker container

You can pre-build the image with all required packages.

Both approaches work depending on the environment.

4. Deploy the Model in Nuclio

Steps:

Open Nuclio dashboard (port usually 8070)

Create a new function

Upload:

main.py

function.yaml

Class file

Mount the model weight file inside the container

Deploy the function

Nuclio will start a serverless API endpoint used for inference.

5. Connect Nuclio to CVAT (Auto-Annotation Integration)

Inside CVAT:

Go to Settings → Machine Learning Models

Add a new model

Enter:

Model name

Nuclio endpoint URL

Class names

Save & Refresh

CVAT will detect the Nuclio function as an ML model.

Now you can test auto-annotation inside CVAT tasks.

6. Prediction Results: CVAT vs Script

It is normal for the results to differ slightly:

Why results differ:

Nuclio uses JPEG decoding → small differences

CVAT resizes images internally

Nuclio may compress the HTTP image

Preprocessing steps may differ

The script environment vs container environment is not identical

CVAT applies postprocessing thresholds

So prediction values are similar but not exactly the same.

This behavior is normal in production ML pipelines.
