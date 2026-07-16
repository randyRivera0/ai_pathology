"""NiceGUI prototype for the LC25000 five-class workflow.

This module intentionally uses mock preprocessing and prediction behavior. It
must not be used for clinical diagnosis or decision-making.
"""

import asyncio
import base64
from dataclasses import dataclass

from nicegui import events, ui


LC25000_CLASSES = (
    "Colon adenocarcinoma",
    "Benign colon tissue",
    "Lung adenocarcinoma",
    "Lung squamous cell carcinoma",
    "Benign lung tissue",
)
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass
class WorkflowState:
    """Track the transient state of the frontend-only workflow."""

    filename: str = ""
    content_type: str = ""
    image_bytes: bytes = b""
    preprocessed: bool = False


state = WorkflowState()


def reset_result() -> None:
    """Hide stale results when the selected image or workflow state changes."""

    result_card.visible = False
    prediction_button.disable()


async def handle_upload(event: events.UploadEventArguments) -> None:
    """Validate an uploaded image and display an in-memory preview."""

    content_type = event.file.content_type.lower()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        ui.notify("Use a PNG or JPEG image.", type="negative")
        return

    image_bytes = await event.file.read()
    if not image_bytes:
        ui.notify("The selected image is empty or unreadable.", type="negative")
        return

    state.filename = event.file.name
    state.content_type = content_type
    state.image_bytes = image_bytes
    state.preprocessed = False

    encoded = base64.b64encode(image_bytes).decode("ascii")
    preview.set_source(f"data:{content_type};base64,{encoded}")
    preview.visible = True
    filename_label.set_text(state.filename)
    status_label.set_text("Image selected — ready for mock preprocessing")
    preprocess_button.enable()
    reset_result()


def handle_rejected_upload() -> None:
    """Tell the user why the upload control rejected a file."""

    ui.notify("Upload one PNG or JPEG image up to 10 MB.", type="negative")


def mock_preprocess() -> None:
    """Advance the prototype to its ready state without transforming pixels."""

    state.preprocessed = True
    status_label.set_text("Mock preprocessing complete — ready to predict")
    prediction_button.enable()
    result_card.visible = False
    ui.notify("Preprocessing simulated", type="positive")


async def mock_predict() -> None:
    """Display a deterministic mock result without loading an ML model."""

    if not state.preprocessed:
        return

    prediction_button.disable()
    preprocess_button.disable()
    status_label.set_text("Running mock inference…")
    await asyncio.sleep(0.8)

    class_index = sum(state.image_bytes[:256]) % len(LC25000_CLASSES)
    confidence = 0.80 + (sum(state.image_bytes[-128:]) % 1900) / 10_000
    predicted_class.set_text(LC25000_CLASSES[class_index])
    confidence_label.set_text(f"{confidence:.1%}")
    status_label.set_text("Mock prediction complete")
    result_card.visible = True
    prediction_button.enable()
    preprocess_button.enable()


ui.page_title("LC25000 Classifier Prototype")

with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-6"):
    with ui.column().classes("gap-1"):
        ui.label("LC25000 Tissue Classifier").classes("text-3xl font-bold")
        ui.label("Five-class histopathology workflow prototype").classes(
            "text-base text-slate-600"
        )

    with ui.card().classes("w-full bg-amber-50 border border-amber-200 shadow-none"):
        ui.label("Research and educational use only").classes("font-semibold text-amber-900")
        ui.label(
            "This prototype is not a medical device and must not be used for "
            "diagnosis or clinical decision-making."
        ).classes("text-amber-800")

    with ui.row().classes("w-full items-stretch gap-6"):
        with ui.card().classes("grow min-w-80"):
            ui.label("1. Select tissue image").classes("text-xl font-semibold")
            ui.upload(
                label="Upload PNG or JPEG",
                on_upload=handle_upload,
                on_rejected=handle_rejected_upload,
                auto_upload=True,
                max_file_size=MAX_FILE_SIZE_BYTES,
                max_files=1,
            ).props("accept=.png,.jpg,.jpeg").classes("w-full")
            preview = ui.image().classes("w-full max-h-80 object-contain rounded")
            preview.visible = False
            filename_label = ui.label("No image selected").classes("text-sm text-slate-600")

        with ui.card().classes("grow min-w-80"):
            ui.label("2. Run workflow").classes("text-xl font-semibold")
            ui.label("Prototype model").classes("text-xs uppercase text-slate-500")
            ui.label("EfficientNetB0 · mock-v1").classes("font-medium")
            ui.separator()
            status_label = ui.label("Waiting for an image").classes("text-slate-600")
            with ui.row().classes("gap-3"):
                preprocess_button = ui.button(
                    "Preprocess", icon="tune", on_click=mock_preprocess
                )
                prediction_button = ui.button(
                    "Predict", icon="science", on_click=mock_predict
                )
            preprocess_button.disable()
            prediction_button.disable()

    with ui.card().classes("w-full") as result_card:
        ui.label("Mock prediction result").classes("text-xl font-semibold")
        with ui.row().classes("w-full gap-12"):
            with ui.column().classes("gap-0"):
                ui.label("Predicted class").classes("text-xs uppercase text-slate-500")
                predicted_class = ui.label().classes("text-lg font-semibold")
            with ui.column().classes("gap-0"):
                ui.label("Confidence").classes("text-xs uppercase text-slate-500")
                confidence_label = ui.label().classes("text-lg font-semibold")
            with ui.column().classes("gap-0"):
                ui.label("Model metadata").classes("text-xs uppercase text-slate-500")
                ui.label("EfficientNetB0 · mock-v1 · 224×224 input")
        ui.label(
            "Simulated output for interface evaluation; no model inference was performed."
        ).classes("text-sm text-slate-500")
    result_card.visible = False


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="127.0.0.1",
        port=8080,
        title="LC25000 Classifier Prototype",
        reload=False,
    )
