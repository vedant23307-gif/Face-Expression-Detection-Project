#!/usr/bin/env python3
"""
Main Entry Point for Railway / Render / Cloud Hosting
Imports and exposes the FastAPI app instance from python/fastapi_server.py
"""

from python.fastapi_server import app

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
