from typing import Any

from nicegui import binding


@binding.bindable_dataclass
class AppState:
    filename: str = ""
    content_type: str = ""
    tissue: str | None = None
    rgb_result: dict[str, Any] | None = None
    gray_result: dict[str, Any] | None = None
    image_bytes: bytes = b""


app_state = AppState()
