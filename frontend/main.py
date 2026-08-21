"""NiceGUI interface for the binary HISTOPANTUM colorectal research workflow."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

from nicegui import events, ui


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference import InferenceService, InvalidImageError
from backend.models.grayscale_resnet50_toy_classifier import (
    GrayscaleResNet50ToyClassifier,
)
from backend.models.protocol import ModelArtifactError


SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_SPECIMEN_SITE = "Colon / colorectum — supported"
UNSUPPORTED_SPECIMEN_SITE = "Other / unknown — not supported"


@dataclass
class WorkflowState:
    """Track the currently selected image in memory."""

    filename: str = ""
    content_type: str = ""
    image_bytes: bytes = b""
    specimen_site: str | None = None


state = WorkflowState()
inference_service = InferenceService()
grayscale_inference_service = InferenceService(GrayscaleResNet50ToyClassifier())


def reset_result() -> None:
    """Hide results that belong to an earlier selected image."""

    result_card.visible = False
    score_lines.set_text("")
    grayscale_predicted_class.set_text("")
    grayscale_confidence_label.set_text("")
    grayscale_status.set_text("")


def update_prediction_availability() -> None:
    """Enable prediction only when an image and supported site are supplied."""

    if state.image_bytes and state.specimen_site == SUPPORTED_SPECIMEN_SITE:
        prediction_button.enable()
    else:
        prediction_button.disable()


def handle_specimen_site_change(event: events.ValueChangeEventArguments) -> None:
    """Store specimen context and enforce the Experiment 9 organ scope."""

    state.specimen_site = event.value
    reset_result()
    if state.specimen_site == SUPPORTED_SPECIMEN_SITE:
        status_label.set_text(
            "Image selected - ready to predict"
            if state.image_bytes
            else "Waiting for an image"
        )
    else:
        status_label.set_text("Select supported colon/colorectal tissue to continue")
    update_prediction_availability()


async def handle_upload(event: events.UploadEventArguments) -> None:
    """Validate an uploaded image envelope and display an in-memory preview."""

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

    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_source = f"data:{content_type};base64,{encoded}"
    rgb_preview.set_source(image_source)
    grayscale_preview.set_source(image_source)
    preview_row.visible = True
    filename_label.set_text(state.filename)
    status_label.set_text(
        "Image selected - select colon/colorectal tissue to continue"
        if state.specimen_site != SUPPORTED_SPECIMEN_SITE
        else "Image selected - ready to predict"
    )
    reset_result()
    update_prediction_availability()


def handle_rejected_upload() -> None:
    """Tell the user why the upload control rejected a file."""

    ui.notify("Upload one PNG or JPEG image up to 10 MB.", type="negative")


async def predict() -> None:
    """Run real model inference outside the NiceGUI event loop."""

    if not state.image_bytes or state.specimen_site != SUPPORTED_SPECIMEN_SITE:
        ui.notify(
            "Select Colon / colorectum before running inference.",
            type="warning",
        )
        return

    prediction_button.disable()
    result_card.visible = False
    status_label.set_text("Running primary RGB inference...")
    selected_image = state.image_bytes
    selected_site = state.specimen_site

    try:
        result = await asyncio.to_thread(inference_service.predict, selected_image)
    except InvalidImageError as error:
        status_label.set_text("Image validation failed")
        ui.notify(str(error), type="negative")
    except ModelArtifactError as error:
        status_label.set_text("Model unavailable")
        ui.notify(str(error), type="negative", timeout=10_000)
    except Exception:
        logging.exception("Unexpected inference failure")
        status_label.set_text("Inference failed")
        ui.notify(
            "Unexpected inference error. Check the application logs.",
            type="negative",
        )
    else:
        if selected_image != state.image_bytes or selected_site != state.specimen_site:
            status_label.set_text("Input changed - run prediction again")
            return
        predicted_class.set_text(result.predicted_class)
        confidence_label.set_text(f"{result.confidence:.1%}")
        score_lines.set_text(
            "\n".join(
                f"{item.class_name}: {item.score:.2%}" for item in result.scores
            )
        )
        grayscale_status.set_text("Running experimental grayscale model...")
        result_card.visible = True

        try:
            grayscale_result = await asyncio.to_thread(
                grayscale_inference_service.predict,
                selected_image,
            )
        except Exception:
            logging.exception("Experimental grayscale inference failure")
            grayscale_status.set_text(
                "Experimental grayscale result unavailable; primary result is valid."
            )
        else:
            if (
                selected_image != state.image_bytes
                or selected_site != state.specimen_site
            ):
                status_label.set_text("Input changed - run prediction again")
                result_card.visible = False
                return
            grayscale_predicted_class.set_text(grayscale_result.predicted_class)
            grayscale_confidence_label.set_text(
                f"{grayscale_result.confidence:.1%}"
            )
            grayscale_status.set_text(
                "Experiment 10 | grayscale toy model | 224x224 RGB accepted"
            )
        status_label.set_text("Prediction complete")
    finally:
        update_prediction_availability()


ui.page_title("HISTOPANTUM Colorectal Classifier")

with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-6"):
    with ui.column().classes("gap-1"):
        ui.label("HISTOPANTUM Colorectal Tissue Classifier").classes(
            "text-3xl font-bold"
        )
        ui.label("Binary colorectal histopathology research workflow").classes(
            "text-base text-slate-600"
        )

    with ui.card().classes("w-full bg-amber-50 border border-amber-200 shadow-none"):
        ui.label("Research and educational use only").classes(
            "font-semibold text-amber-900"
        )
        ui.label(
            "This application is not a medical device and must not be used for "
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
            with ui.row().classes("w-full gap-4") as preview_row:
                with ui.column().classes("grow min-w-48 gap-1"):
                    ui.label("Original RGB model input").classes(
                        "text-sm font-medium text-slate-700"
                    )
                    rgb_preview = ui.image().classes(
                        "w-full max-h-72 object-contain rounded"
                    )
                with ui.column().classes("grow min-w-48 gap-1"):
                    ui.label("Grayscale visualization").classes(
                        "text-sm font-medium text-slate-700"
                    )
                    grayscale_preview = ui.image().classes(
                        "w-full max-h-72 object-contain rounded"
                    ).style("filter: grayscale(100%);")
            preview_row.visible = False
            ui.label(
                "The grayscale view is a visual comparison only. Experiment 9 "
                "runs inference on the original RGB image and uses its colour "
                "information."
            ).classes("text-xs text-slate-500")
            filename_label = ui.label("No image selected").classes(
                "text-sm text-slate-600"
            )

        with ui.card().classes("grow min-w-80"):
            ui.label("2. Run inference").classes("text-xl font-semibold")
            ui.select(
                options=[SUPPORTED_SPECIMEN_SITE, UNSUPPORTED_SPECIMEN_SITE],
                label="Known specimen site (required)",
                on_change=handle_specimen_site_change,
            ).props("outlined").classes("w-full")
            ui.label(
                "Experiment 9 assumes the image is colorectal tissue; it does "
                "not identify the anatomical origin. Other or unknown sites are "
                "outside the model's validated research scope."
            ).classes("text-sm text-slate-600")
            ui.label("Experiment model").classes("text-xs uppercase text-slate-500")
            ui.label("ResNet50 | exp-9 | final locked model").classes("font-medium")
            ui.separator()
            status_label = ui.label("Waiting for an image").classes("text-slate-600")
            prediction_button = ui.button("Predict", icon="science", on_click=predict)
            prediction_button.disable()

    with ui.card().classes("w-full") as result_card:
        ui.label("Prediction result").classes("text-xl font-semibold")
        with ui.row().classes("w-full gap-12"):
            with ui.column().classes("gap-0"):
                ui.label("Predicted class").classes("text-xs uppercase text-slate-500")
                predicted_class = ui.label().classes("text-lg font-semibold")
            with ui.column().classes("gap-0"):
                ui.label("Model confidence").classes("text-xs uppercase text-slate-500")
                confidence_label = ui.label().classes("text-lg font-semibold")
            with ui.column().classes("gap-0"):
                ui.label("Model metadata").classes("text-xs uppercase text-slate-500")
                ui.label("ResNet50 | exp-9 | 224x224 RGB input")
        ui.label("All class scores").classes("text-sm font-semibold mt-3")
        score_lines = ui.label().classes("text-sm whitespace-pre-line text-slate-700")
        ui.label(
            "Research image-classification output only—not a final pathology "
            "diagnosis. Final interpretation requires specimen provenance, "
            "clinical information, examination of the complete specimen, and "
            "potentially ancillary testing. Confidence is not a calibrated "
            "clinical probability."
        ).classes("text-sm text-slate-500")
        ui.separator().classes("my-3")
        ui.label("Experimental grayscale comparison").classes(
            "text-lg font-semibold text-slate-700"
        )
        with ui.row().classes("w-full gap-12"):
            with ui.column().classes("gap-0"):
                ui.label("Toy-model class").classes(
                    "text-xs uppercase text-slate-500"
                )
                grayscale_predicted_class = ui.label().classes("font-semibold")
            with ui.column().classes("gap-0"):
                ui.label("Toy-model confidence").classes(
                    "text-xs uppercase text-slate-500"
                )
                grayscale_confidence_label = ui.label().classes("font-semibold")
        grayscale_status = ui.label().classes("text-sm text-slate-600")
        ui.label(
            "Experiment 10 is a short grayscale toy-model ablation with a "
            "different training protocol and evaluation split. Its score is "
            "not directly comparable to Experiment 9 and is not a diagnosis."
        ).classes("text-sm text-rose-700")
    result_card.visible = False


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="127.0.0.1",
        port=8080,
        title="HISTOPANTUM Colorectal Classifier",
        reload=False,
    )
