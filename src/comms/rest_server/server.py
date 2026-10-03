import signal
import sys
import os
import click
import uvicorn
from typing import List
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from src.comms.socket.socket_server import asgi
from src.comms.rest_server.api.commands import CameraRoutes
from src.comms.rest_server.api.camera_add import add_CameraRoutes
from src.comms.rest_server.api.jwt_utils import JWT_token
from src.comms.rest_server.api.session import SessionRoutes
from src.comms.rest_server.api.user_registration import User_registration

from src.io_manager.sensors.camera import IP_Camera 

from src.orchestrator.orchestrator import Orchestrator
from src.utils.config import read_base_config, load_config

BASE_CONFIG = read_base_config()
REST_SERVER_CONFIG = load_config(BASE_CONFIG["rest_server"])

# ip_cam=CameraStreamer()

class Server:

    def __init__(self) -> None:
        self.orchestrator = None

    def run_api_server(self, orchestrator: Orchestrator, env: str = "prod", debug: bool = False):
        os.environ["ENV"] = env
        os.environ["DEBUG"] = str(debug)
        self.orchestrator = orchestrator

        self.init_routers(app_=self.app_)
        
        uvicorn.run(
            app=REST_SERVER_CONFIG["app"],
            host=REST_SERVER_CONFIG["host"],
            port=REST_SERVER_CONFIG["port"],
            reload=REST_SERVER_CONFIG["reload"],
            workers=REST_SERVER_CONFIG["workers"],
        )

    def init_routers(self, app_: FastAPI) -> None:

        
        camera_routes = CameraRoutes(self.orchestrator)
        add_cam_routes = add_CameraRoutes(self.orchestrator)
        jwt_token = JWT_token()
        session_routes = SessionRoutes()
        user_registration = User_registration()
        app_.include_router(jwt_token.router)
        app_.include_router(camera_routes.router)
        app_.include_router(add_cam_routes.router)
        app_.include_router(session_routes.router)
        app_.include_router(user_registration.router)
        app_.add_websocket_route("/socket.io/", asgi) 
        
        # # added routes for start and stop streaming
        # app_.add_api_route("/camera/start/", self.start_camera_streaming, methods=["GET"])
        # app_.add_api_route("/camera/stop/", self.stop_camera_streaming, methods=["GET"])


    def make_middleware(self) -> List[Middleware]:
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=REST_SERVER_CONFIG["allow_origins"],
                allow_credentials=REST_SERVER_CONFIG["allow_credentials"],
                allow_methods=REST_SERVER_CONFIG["allow_methods"],
                allow_headers=REST_SERVER_CONFIG["allow_headers"],
            ),
        ]
        return middleware

    def create_app(self) -> FastAPI:
        self.app_ = FastAPI(
            title=REST_SERVER_CONFIG["title"],
            description=REST_SERVER_CONFIG["description"],
            version=REST_SERVER_CONFIG["version"],
            docs_url=REST_SERVER_CONFIG["docs_url"],
            redoc_url=REST_SERVER_CONFIG["redoc_url"],
            middleware=self.make_middleware(),
        )
        return self.app_


server = Server()
app = server.create_app()


# Necessary to kill the server whenever Interrupt signal is sent
def receive_signal(signalNumber, frame):
    print('Received:', signalNumber)
    sys.exit()

@app.on_event("startup")
async def startup_event():
    signal.signal(signal.SIGINT, receive_signal)

