# AI Pathology

Research and educational prototype for classifying LC25000 histopathology
images. The current demo performs binary classification of benign colon tissue
and colon adenocarcinoma with the final group-aware ResNet50 model.

> This project is not a medical device and must not be used for diagnosis or
> clinical decision-making. Model confidence is not a calibrated clinical
> probability.

## Current architecture

The demo is a layered modular monolith:

```text
NiceGUI -> InferenceService -> ResNet50 adapter -> TensorFlow/Keras checkpoint
```

The framework-neutral adapter contract keeps the frontend independent of
TensorFlow. The earlier five-class Experiment 1 adapter remains available in
the repository but is detached from the default demo. The current UI is
intended for a single local user.

## Repository structure

- `frontend/`: NiceGUI prototype application.
- `backend/`: inference orchestration and independently managed models.
- `experiments/`: historical training and evaluation experiments.
- `notebooks/final/`: executed final notebook and compact evaluation evidence.

## Run locally (Windows PowerShell)

Python 3.13 is the verified combined NiceGUI and TensorFlow runtime. Create the
dedicated model environment from the repository root:

```powershell
py -3.13 -m venv .venv-model
.\.venv-model\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the verified final artifact at
`notebooks/final/resnet50_binary_best.keras`, or point to it with an environment
variable:

```powershell
$env:AI_PATHOLOGY_RESNET50_MODEL_PATH = "C:\path\to\resnet50_binary_best.keras"
```

The application requires SHA-256
`e48e3afcf2521e731fa68ed4f8585cb3c19d864f5d563aae74246cd88a4d86d1`
and validates the model input and output shapes before inference. Model weights
are intentionally excluded from ordinary Git history.

Run the application from the repository root:

```powershell
python frontend\main.py
```

Open `http://127.0.0.1:8080` if a browser does not open automatically. Stop the
server with `Ctrl+C`.

The demo accepts one PNG or JPEG image at a time and reports ordered benign and
adenocarcinoma scores. These scores are research outputs, not calibrated
clinical probabilities.
