import asyncio

from src.utils.config import read_base_config, load_config

BASE_CONFIG = read_base_config()
STATUS_BAR_STATE = load_config(BASE_CONFIG["status_bar_state"])


class SocketConnection:

    socket_ = None
    last_state_info = None

    @classmethod
    def send_status_update(cls, key):

        status_info = STATUS_BAR_STATE[key]

        if cls.socket_ is not None and status_info != cls.last_state_info:
            # print("sending from socketio:", status_info)
            asyncio.run(cls.socket_.emit(
                                    "status_update",
                                    {
                                        "message": status_info["text"],
                                        "color": status_info["color"] ,
                                    },
                                    ))

            cls.last_state_info = status_info

    @classmethod
    def send_results_update(cls, dfmd_res):

        if cls.socket_ is not None:
            asyncio.run(cls.socket_.emit(
                "get_results", 
                {"data": dfmd_res}))
            
            print("sending from sockio", dfmd_res)
