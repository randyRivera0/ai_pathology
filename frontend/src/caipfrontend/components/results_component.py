from caipfrontend.state import app_state
from nicegui import ui


def dict_preprocessing_score_lines(v: list):

    return "\n".join(f"{item['class_name']}: {item['score']:.2%}" for item in v)


def results_component(target_name: str):
    with ui.row().classes("w-full gap-12"):
        with ui.column().classes("gap-0"):
            ui.label("Predicted class").classes("text-xs uppercase text-slate-500")
            ui.label().classes("text-lg font-semibold").bind_text_from(
                app_state, (target_name, "predicted_class"), strict=False, backward=str
            )
        with ui.column().classes("gap-0"):
            ui.label("Model confidence").classes("text-xs uppercase text-slate-500")
            ui.label().classes("text-lg font-semibold").bind_text_from(
                app_state, (target_name, "confidence"), strict=False, backward=str
            )
        with ui.column().classes("gap-0"):
            ui.label("Model metadata").classes("text-xs uppercase text-slate-500")
            ui.label().bind_text_from(
                app_state, (target_name, "metadata"), strict=False, backward=str
            )
    ui.label("All class scores").classes("text-sm font-semibold mt-3")
    ui.label().classes("text-sm whitespace-pre-line text-slate-700").bind_text_from(
        app_state, (target_name, "scores"), strict=False, backward=dict_preprocessing_score_lines
    )
