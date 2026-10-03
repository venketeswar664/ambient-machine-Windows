
from datetime import datetime, timedelta, timezone
from typing import Optional
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status, Query, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from classy_fastapi import Routable, get,post

from src.database.schemas.users_schema import Users

import secrets
import time
import base64

# to generate a secret key, you can use a tool like openssl rand -hex 32
SECRET_KEY = "a365714e77efacceaf8e0a1708fe1a8d522ad3b45ded10b5ee40dde603cf432e"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5
REFRESH_TOKEN_EXPIRE_MINUTES = 5

fake_users_db = {
    "neophyte": {
        "username": "neophyte",
        "full_name": "Neophyte",
        "email": "",
        "hashed_password": "$2b$12$FlInq/spxnDZr3IYGXCzbed7JjO9Ty1xAkta.dE5yJ1ry4wBCMwN2",
        "disabled": False,
    },
    "digital": {
        "username": "digital",
        "full_name": "Digital",
        "email": "",
        "hashed_password": "$2b$12$p/uDEprooahmQJJ6YlH4e.lvLkC0ONjJ.Ci020e/qZzt6ZsCZ7OR2",
        "disabled": False,
    }
}

# TODO : Fetch this users db from MongodDB



class Token(BaseModel):
    access_token: str
    token_type: str
    status_codess: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")
blacklisted_tokens = set()
user_sessions = {}

class JWT_token(Routable):
    blacklisted_tokens = set()

    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)


    def get_password_hash(self, password):
        return pwd_context.hash(password)


    def get_user(self, db, username: str):
        if username in db:
            user_dict = db[username]
            return UserInDB(**user_dict)


    def authenticate_user(self, fake_db, username: str, password: str):
        user = self.get_user(fake_db, username)
        if not user:
            return False
        if not self.verify_password(password, user.hashed_password):
            return False
        return user


    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=60)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt


    async def get_current_user(self, token: Annotated[str, Depends(oauth2_scheme)]):
        
        jwt_token_instance = JWT_token()
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if token in blacklisted_tokens:
            return {"message": "Invalid Token"}
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
        except JWTError:
            raise credentials_exception
        user = jwt_token_instance.get_user(fake_users_db, username=token_data.username)
        if user is None:
            raise credentials_exception
        return user
    

    def create_refresh_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def get_all_users_from_db(self):
        try:
            if self.db_users_dict == {}:
                return self.db_users_dict
        except: 
            print("Exception while reading db users dict")
        
        db_users = Users.get_all()
        self.db_users_dict = {}
        for dbuser in db_users:
            self.db_users_dict[dbuser.username] = dbuser._data

        print(f"All DB db_users_dict: {self.db_users_dict}")
        return self.db_users_dict

    @post("/user/login")
    async def login_for_access_token(self, 
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],  request: Request
    ) -> Token:
        username = form_data.username
        password = form_data.password
        db_users_dict = self.get_all_users_from_db()

        print(f"Username password: {username,password}, {type(username)}, {type(password)}")

        username = base64.b64decode(form_data.username).decode('utf-8')
        password = base64.b64decode(form_data.password).decode('utf-8')
        print(f"Decoded username password: {username,password}, {type(username)}, {type(password)}")


        user = self.authenticate_user(db_users_dict, username, password)
        print(f"User user: {user}")

        client_ip = request.client.host
        print(f"Client IP: {client_ip}")

        db_users_dict[username]["disabled"] = False
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password are required",
            )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,               
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token_expires = None
        refresh_token_expires = timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
        refresh_token = self.create_refresh_token(data={"sub": user.username}, expires_delta=refresh_token_expires)
        return Token(access_token=access_token, refresh_token=refresh_token, token_type="bearer", status_codess="200")
    # def create_session_token(self, token: str):
    #     """
    #     Create a session token for the given user.
    #     """
    #     credentials_exception = HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    #     if token in blacklisted_tokens:
    #         raise credentials_exception
    #     try:
    #         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #         username = payload.get("sub")
            
    #         session_token = secrets.token_urlsafe(32)
    #         session_start_time = time.time()
    #         user_sessions[session_token] = {"username": username, "start_time": session_start_time}
    #         return {"message: Session created successfully"}
    #     except JWTError:
    #         return {"message: Session could not start"}


    # @post("/start_session")
    # async def start_session(self, token: str, current_user: User = Depends(get_current_user)):
    #     credentials_exception = HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    #     """
    #     if token doesnt exist raise exception 
    #     """
    #     if token in blacklisted_tokens:
    #         raise credentials_exception  
        
    #     session_token = self.create_session_token(token)           
            
    #     # try:
    #     #     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #     #     session_token = payload.get("session_token")
    #     #     if session_token is None or session_token not in user_sessions:
    #     #         raise credentials_exception
    #     # #     username: str = payload.get("sub")
    #     # #     if username is None:
    #     # #         raise credentials_exception
            
    #     # except JWTError:
    #     #     raise credentials_exception
    #     """
    #     If token is valid then session is created successfully then this is the response in which we 
    #     have session token and message as session started
    #     """
    #     return current_user, { "session_token": session_token, "message": "Session Started"}

    # @post("/logout")
    # async def logout(self, token: str ):
    #     if token in blacklisted_tokens:
    #         return {"message": "Logged out not successfully"}
    #     else:
    #        # Decode the token
    #         decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #         username = decoded_token.get("sub")
    #         if username:
    #             # Disable user in the fake database
    #             fake_users_db[username]["disabled"] = True
                
    #         session_token = decoded_token.get("session_token")
    #         session_duration = 0
    #         if session_token:
    #             session_info = user_sessions.get(session_token)
    #             if session_info:
    #                 session_start_time = session_info["start_time"]
    #                 session_duration = time.time() - session_start_time
    #                 user_sessions.pop(session_token)
                    
    #         # Update the expiration time for the given active token
    #         new_exp = datetime.now(timezone.utc) + timedelta(seconds=1)
    #         decoded_token["exp"] = new_exp

    #         # Re-encode the token with the updated expiration time
    #         new_token = jwt.encode(decoded_token, SECRET_KEY, algorithm=ALGORITHM)
    #         blacklisted_tokens.add(token)
    #         return {"message": f"Logged out successfully. Session duration: {session_duration} seconds."}      
            

    @post("/refresh-token")
    async def refresh_access_token(self, refresh_token: str
    ) -> Token:
            try:
                payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
                username: str = payload.get("sub")
                if username is None:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
                user = self.get_user(fake_users_db, username=username)
                if user is None:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                access_token = self.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
                return Token(access_token=access_token, token_type="bearer")
            except JWTError:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
       

if __name__=="__main__":
    user = JWT_token().authenticate_user(fake_users_db,username="digital", password="Digital")
    print(f"User user: {user}")
