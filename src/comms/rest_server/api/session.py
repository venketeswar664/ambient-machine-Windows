
from classy_fastapi import Routable, get
from fastapi.responses import StreamingResponse
from src.orchestrator.orchestrator import Orchestrator
from src.io_manager.sensors.camera import IP_Camera
from src.comms.rest_server.api.jwt_utils import JWT_token, User
from fastapi import Depends

from fastapi import Depends, FastAPI, HTTPException, status, Query
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from classy_fastapi import Routable, get,post
import secrets
import time

blacklisted_tokens = set()
user_sessions = {}

SECRET_KEY = "a365714e77efacceaf8e0a1708fe1a8d522ad3b45ded10b5ee40dde603cf432e"
ALGORITHM = "HS256"

class SessionRoutes(Routable):


    def create_session_token(self, token: str):
        """
        Create a session token for the given user.
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if token in blacklisted_tokens:
            raise credentials_exception
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            
            session_token = secrets.token_urlsafe(32)
            session_start_time = time.time()
            user_sessions[session_token] = {"username": username, "start_time": session_start_time}
            return {"message: Session created successfully"}
        except JWTError:
            raise credentials_exception
        
    @post("/start_session")
    async def start_session(self, token:str, current_user: User = Depends(JWT_token.get_current_user)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        """
        if token doesnt exist raise exception 
        """
        if token in blacklisted_tokens:
            raise credentials_exception  
        
        session_token = self.create_session_token(token)           
            
        # try:
        #     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        #     session_token = payload.get("session_token")
        #     if session_token is None or session_token not in user_sessions:
        #         raise credentials_exception
        # #     username: str = payload.get("sub")
        # #     if username is None:
        # #         raise credentials_exception
            
        # except JWTError:
        #     raise credentials_exception
        """
        If token is valid then session is created successfully then this is the response in which we 
        have session token and message as session started
        """
        return { "session_token": session_token, "message": "Session Started"}


    @post("/logout")
    async def logout(self, token: str , current_user: User = Depends(JWT_token.get_current_user)):
        if token in blacklisted_tokens:
            return {"message": "Logged out not successfully"}
        else:
           # Decode the token
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            session_token = decoded_token.get("session_token")
            session_duration = 0
            if session_token:
                session_info = user_sessions.get(session_token)
                if session_info:
                    session_start_time = session_info["start_time"]
                    session_duration = time.time() - session_start_time
                    user_sessions.pop(session_token)
                    
            # Update the expiration time for the given active token
            new_exp = datetime.now(timezone.utc) + timedelta(seconds=1)
            decoded_token["exp"] = new_exp

            # Re-encode the token with the updated expiration time
            new_token = jwt.encode(decoded_token, SECRET_KEY, algorithm=ALGORITHM)
            blacklisted_tokens.add(token)
            return {"message": f"Logged out successfully. Session duration: {session_duration} seconds."}      
            
