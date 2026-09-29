from typing import Any

from caipfrontend.session import session


async def inference(
    url: str, filename: str, content_type: str, image_bytes: bytes, tissue: str
) -> dict[str, Any]:
    files = {"file": (filename, image_bytes, content_type)}
    data = {"tissue": tissue}
    response = await session.post(url=url, files=files, data=data)
    response.raise_for_status()
    return response.json()
