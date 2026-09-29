import os

from dotenv import load_dotenv
import httpx

load_dotenv()

backend_api_url: str = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

session = httpx.AsyncClient(base_url=str(backend_api_url), timeout=httpx.Timeout(60.0, connect=5.0))
