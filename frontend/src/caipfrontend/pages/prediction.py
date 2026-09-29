import base64

from caipfrontend.components.results_component import results_component
from caipfrontend.hooks.predict import handle_predict
from caipfrontend.state import app_state
from nicegui import events, ui

SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_SPECIMEN_SITE = "Colon"
UNSUPPORTED_SPECIMEN_SITE = "Other / unknown — not supported"


def preprocess_bytes_to_base64(value):
    encoded = base64.b64encode(value).decode("ascii")
    return f"data:{app_state.content_type};base64,{encoded}"


async def handle_upload(event: events.UploadEventArguments):
    app_state.filename = event.file.name
    app_state.content_type = event.file.content_type
    app_state.image_bytes = await event.file.read()


def handle_rejected_upload() -> None:
    """Tell the user why the upload control rejected a file."""

    ui.notify("Upload one PNG or JPEG image up to 10 MB.", type="negative")


def prediction_page():
    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-6"):
        # Title
        ui.label("AI pathology chat").classes("text-3xl font-bold")
        # Disclaimer
        with ui.card().classes("w-full bg-amber-50 border border-amber-200 shadow-none"):
            ui.label("Research and educational use only").classes("font-semibold text-amber-900")
            ui.label(
                "This application is not a medical device and must not be used for "
                "diagnosis or clinical decision-making."
            ).classes("text-amber-800")
        # Configuration
        with ui.row().classes("w-full items-stretch gap-6"):
            # Section 1: tissue selection
            with ui.card().classes("grow min-w-80"):
                ui.label("1. Select tissue").classes("text-xl font-semibold")
                ui.select(
                    options=[SUPPORTED_SPECIMEN_SITE, UNSUPPORTED_SPECIMEN_SITE],
                    label="Known specimen site (required)",
                ).props("outlined").classes("w-full").bind_value(app_state, "tissue")
                ui.label(
                    "Experiment 9 assumes the image is colorectal tissue; it does "
                    "not identify the anatomical origin. Other or unknown sites are "
                    "outside the model's validated research scope."
                ).classes("text-sm text-slate-600")
            # Section 2: image upload
            with ui.card().classes("grow min-w-80"):
                ui.label("2. Select tissue image").classes("text-xl font-semibold")
                ui.upload(
                    label="Upload tissue image",
                    on_upload=handle_upload,
                    on_rejected=handle_rejected_upload,
                    auto_upload=True,
                    max_file_size=MAX_FILE_SIZE_BYTES,
                    max_files=1,
                ).props("accept=.png,.jpg,.jpeg").classes("w-full").bind_enabled_from(
                    target_object=app_state, target_name="tissue", backward=lambda v: v == "Colon"
                )
                # Preview Section
                with ui.row().classes("w-full gap-4") as preview_row:
                    with ui.column().classes("grow min-w-48 gap-1"):
                        ui.label("Original RGB model input").classes(
                            "text-sm font-medium text-slate-700"
                        )
                        ui.image().classes(
                            "w-full max-h-72 object-contain rounded"
                        ).bind_source_from(
                            app_state, "image_bytes", backward=preprocess_bytes_to_base64
                        )
                    with ui.column().classes("grow min-w-48 gap-1"):
                        ui.label("Grayscale visualization").classes(
                            "text-sm font-medium text-slate-700"
                        )
                        ui.image().classes("w-full max-h-72 object-contain rounded").style(
                            "filter: grayscale(100%);"
                        ).bind_source_from(
                            app_state, "image_bytes", backward=preprocess_bytes_to_base64
                        )
                        """
                        ui.label(
                                                    "The grayscale view is a visual comparison only. Experiment 9 "
                                                    "runs inference on the original RGB image and uses its colour "
                                                    "information."
                                                ).classes("text-xs text-slate-500")
                        """
                preview_row.bind_visibility_from(app_state, "image_bytes", lambda v: v != b"")

        ui.button(text="Predict", icon="science", on_click=handle_predict).bind_enabled_from(
            target_object=app_state, target_name="tissue", backward=lambda v: v == "Colon"
        )
        # Results
        with ui.card().classes("w-full") as result_card:
            ui.label("Prediction result").classes("text-xl font-semibold")
            # RGB predictions
            results_component(target_name="rgb_result")
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
            # Grayscale predictions
            results_component(target_name="gray_result")
            ui.label(
                "Experiment 10 is a short grayscale toy-model ablation with a "
                "different training protocol and evaluation split. Its score is "
                "not directly comparable to Experiment 9 and is not a diagnosis."
            ).classes("text-sm text-rose-700")
        result_card.bind_visibility_from(app_state, "rgb_result")
