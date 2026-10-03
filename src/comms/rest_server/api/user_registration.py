# -*- coding: utf-8 -*-
'''
Created on 2024-08-06 14:16
Project: ambient-machine
@author: Aryan Sinha
@email: arison.aryan@gmail.com
'''


from fastapi import APIRouter, HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from src.core.machine_learning.computer_vision.face_registration import FaceRegistration
from classy_fastapi import Routable, get, post
from sse_starlette.sse import EventSourceResponse
import asyncio

# Define a Pydantic model for the request body
class UserRegistrationRequest(BaseModel):
    newusername: str
    firstname: str
    lastname: str
    phonenumber: str
    role: str
    newpassword: str
    email: str
    


# Initialize UserRegistration instance
class User_registration(Routable):
    def __init__(self):
        super().__init__()
        self.name = None
        self.phone_number = None
        self.position = None
        self.user_registration = None
        self.streaming_active = False

    @post("/register_user")
    async def register_user(self, request: UserRegistrationRequest):
        self.user_registration = FaceRegistration()
        self.username = request.newusername
        self.name = f"{request.firstname}_{request.lastname}"
        self.phone_number = request.phonenumber
        self.position = request.role
        print(request)

        if not (self.name and self.phone_number and self.position):
            raise HTTPException(status_code=400, detail="Name, phone number, and position are all required.")
        
        print("Config set. Ready for Capturing faces...")
        embeddings_path = self.user_registration.set_user_details(request)
        return {"message": "Config set. Ready for Capturing faces... ",
                "embeddings_path": embeddings_path
                }
    
    @get("/preview_user")
    async def preview_user(self):
        if not (self.name and self.phone_number):
            raise HTTPException(status_code=400, detail="User information is not set. Please register first.")
        self.streaming_active = True

        async def event_generator():
            try:
                async for frame in self.user_registration.register_user_from_camera(self.name, self.phone_number):
                    if not self.streaming_active:
                        break
                    yield frame
                    if self.user_registration.is_registration_complete():
                        self.streaming_active = False
                        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nEOF\r\n'
                        break
            except Exception as e:
                print(f"Error in event_generator: {e}")
                self.streaming_active = False

        return StreamingResponse(event_generator(), media_type='multipart/x-mixed-replace; boundary=frame')
    
    @get("/streaming_status")
    async def streaming_status(self):
        async def event_generator():
            while True:
                if not self.streaming_active:
                    yield {"event": "status", "data": "Streaming stopped"}
                    break
                await asyncio.sleep(1)

        return EventSourceResponse(event_generator())
    

    
    