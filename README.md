# AI Pathology

Research and educational prototype for classifying LC25000 histopathology
images across all five dataset classes. The current demo runs genuine local
inference with the EfficientNetB7 artifact recovered from Experiment 1.

> This project is not a medical device and must not be used for diagnosis or
> clinical decision-making. Model confidence is not a calibrated clinical
> probability.

## Current architecture

The demo is a layered modular monolith:

```text
NiceGUI -> InferenceService -> model adapter -> TensorFlow/Keras H5 artifact
```

The framework-neutral adapter contract allows later models to be introduced
without coupling the frontend to TensorFlow. The current UI is intended for a
single local user.

## Repository structure

- `frontend/`: NiceGUI prototype application.
- `backend/`: inference orchestration and independently managed models.
- `experiments/`: training and recovery/evaluation notebooks.
## Run locally (Windows PowerShell)

Python 3.13 is the verified combined GUI and TensorFlow runtime.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the verified Experiment 1 artifact at `models/model.h5`, or point to it
with an environment variable:

```powershell
$env:AI_PATHOLOGY_MODEL_PATH = "C:\path\to\model.h5"
```

The application verifies the artifact SHA-256 before loading it. Model weights
are intentionally excluded from Git.

Run the application from the repository root:

```powershell
python frontend\main.py
```

Open `http://localhost:8080` if a browser does not open automatically. Stop the
server with `Ctrl+C`.
