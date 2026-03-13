import asyncio
import cv2
import base64
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Set

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.settings = {"show_wireframe": False, "device": "cpu"}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_frame(self, frame, metadata: dict = None):
        if not self.active_connections:
            return

        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            logger.error("Failed to encode frame")
            return

        # Convert to base64
        b64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        payload = {
            "type": "frame",
            "image": f"data:image/jpeg;base64,{b64_image}",
        }
        
        if metadata:
            payload.update(metadata)

        # Broadcast to all connected clients
        message = json.dumps(payload)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                self.disconnect(connection)

# Create FastAPI and Manager instances
app = FastAPI(title="ChibiCam Stream Server")
manager = ConnectionManager()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We can also receive commands from the React frontend here
            data_str = await websocket.receive_text()
            try:
                data = json.loads(data_str)
                # Handle incoming commands (e.g., change settings)
                if data.get("type") == "settings":
                    if "viewMode" in data:
                        manager.settings["view_mode"] = data["viewMode"]
                    if "currentFilter" in data:
                        manager.settings["currentFilter"] = data["currentFilter"]
                    if "pookieMode" in data:
                        manager.settings["pookieMode"] = data["pookieMode"]
                    if "showWireframe" in data:
                        manager.settings["show_wireframe"] = data["showWireframe"]
                    if "useGpu" in data:
                        manager.settings["device"] = "cuda" if data["useGpu"] else "cpu"
                    logger.info(f"Received command: {data}")
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

import os
# Serve frontend if available
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")
else:
    logger.warning("Frontend 'dist' directory not found. Serving API only.")
