# AI Pathology

Research and educational prototype for classifying LC25000 histopathology
images across all five dataset classes. This project is not a medical device
and must not be used for diagnosis or clinical decision-making.

Agentic AI assistants contributing to this repository must comply with the
project policies in `AGENTS.md`.

## Repository structure

- `frontend/`: NiceGUI prototype application.
- `backend/`: future inference services and independently managed models.
- `research/`: experimental notebooks.
- `docs/`: paper outcomes, design decisions and references; protected from agent edits.

## Frontend environment (Windows PowerShell)

Python 3.14.3 and NiceGUI 3.14.0 are currently verified for the GUI prototype.

Create and activate the local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the declared dependency:

```powershell
python -m pip install -r requirements.txt
```

Run the frontend from the repository root:

```powershell
python frontend\main.py
```

Open `http://localhost:8080` if the browser does not open automatically. Stop
the server with `Ctrl+C`. The current prediction and preprocessing behavior is
mocked; no ML model is loaded.

To leave the environment:

```powershell
deactivate
```
