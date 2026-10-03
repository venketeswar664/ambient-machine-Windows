import socketio
from src.comms.socket.socket_connection import SocketConnection
from src.utils.config import read_base_config, load_config

BASE_CONFIG = read_base_config()
STATUS_BAR_STATE = load_config(BASE_CONFIG["status_bar_state"])

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins='*')
asgi = socketio.ASGIApp(socketio_server=sio, socketio_path="socket.io")



@sio.event
async def connect(sid, env, auth):
    print("Trying to connect with socket id", sid)


@sio.on("message")
async def handle_message(sid, data):
    print(f"Socket connected: {data}")
    SocketConnection.socket_ = sio
    # SocketConnection.send_status_update("connect")
    status_info = STATUS_BAR_STATE["connect"]
    await sio.emit(
                    "status_update",
                    {
                        "message": status_info["text"],
                        "color": status_info["color"] ,
                    },
                    )
   