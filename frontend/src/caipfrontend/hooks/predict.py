import httpx
from caipfrontend.services.ai_service import inference
from caipfrontend.state import app_state


async def handle_predict():
    try:
        response = await inference(
            url="/predict/rgb",
            filename=app_state.filename,
            content_type=app_state.content_type,
            image_bytes=app_state.image_bytes,
            tissue=app_state.tissue,
        )
        app_state.rgb_result = response
    except httpx.TimeoutException:
        app_state.rgb_result = "The backend request timed out."
    except httpx.HTTPStatusError as exc:
        app_state.rgb_result = f"Backend returned HTTP {exc.response.status_code}."
    except httpx.RequestError:
        app_state.rgb_result = "Cannot reach the backend. Check that the API is running."
    except ValueError:
        app_state.rgb_result = "The backend returned an invalid JSON response."

    try:
        response = await inference(
            url="/predict/gray",
            filename=app_state.filename,
            content_type=app_state.content_type,
            image_bytes=app_state.image_bytes,
            tissue=app_state.tissue,
        )
        app_state.gray_result = response
    except httpx.TimeoutException:
        app_state.gray_result = "The backend request timed out."
    except httpx.HTTPStatusError as exc:
        app_state.gray_result = f"Backend returned HTTP {exc.response.status_code}."
    except httpx.RequestError:
        app_state.gray_result = "Cannot reach the backend. Check that the API is running."
    except ValueError:
        app_state.gray_result = "The backend returned an invalid JSON response."
