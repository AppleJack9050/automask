import os
import sqlite3
from uuid import uuid4
import time
from fastapi.security import OAuth2PasswordBearer # type: ignore
from pwdlib import PasswordHash # type: ignore
import jwt # type: ignore
import secrets
import hmac

USERS_DB = os.getenv("USERS_DB")
SECRET_KEY = secrets.token_hex(32)

class UserManager():
    def __init__(self, session_timeout = 1800):
        self.init_user_table()
        self.password_hash = PasswordHash.recommended()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.algorithm = "HS256"
        self.session_timeout = session_timeout

    def init_user_table(self):
        with sqlite3.connect(USERS_DB) as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                user_name TEXT,
                stored_password TEXT,
                current_token TEXT,
                token_timestamp BIGIINT
            )
            """)
            c.commit()        

    def create_user(self, user_name: str, password: str) -> dict | None:
        if not self.__user_exists(user_name):
            timestamp = time.time()
            token = self.__generate_jwt_token({'bearer':user_name, 'start_time':timestamp})
            hashed_password = self.__hash_user_password(password)

            with sqlite3.connect(USERS_DB) as c:
                c.execute(
                    """
                        INSERT INTO users (id, user_id, user_name, stored_password, current_token, token_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        str(uuid4()),
                        user_name,
                        hashed_password,
                        token,
                        timestamp
                    ))
                c.commit()

                return {"username":user_name, "token":token}

            return None

    def get_token(self, user_name, password) -> tuple | None:
        if self.__user_exists(user_name):
            stored_password = self.__fetch_password(user_name)
            if self.__verifiy_user_password(password, stored_password):
                return self.__fetch_token(user_name)

        return None

    def update_user_name(self, old_user_name: str, new_username: str, password: str) -> dict | None:
        if self.__verifiy_user_password(password, self.__fetch_password(old_user_name)) and  not self.__user_exists(new_username):
            timestamp = time.time()
            token = self.__generate_jwt_token({'bearer':new_username, 'start_time':timestamp})
            with sqlite3.connect(USERS_DB) as c:
                q = c.execute(    
                    """
                        UPDATE users
                        SET user_name = ?, current_token = ?, token_timestamp = ?
                        WHERE user_name = ?
                    """,
                    (new_username, token, timestamp, old_user_name))
                c.commit()
            return {"username":new_username, "token":token}
        else:
            raise PermissionError("Invalid credentials")

    def update_user_password(self, user_name: str, old_password: str, new_password: str) -> dict | None:
        if self.__verifiy_user_password(old_password, self.__fetch_password(user_name)):
            new_password = self.__hash_user_password(new_password)
            timestamp = time.time()
            token = self.__generate_jwt_token({'bearer':user_name, 'start_time':timestamp})
            with sqlite3.connect(USERS_DB) as c:
                q = c.execute(    
                    """
                        UPDATE users
                        SET stored_password = ?, current_token = ?, token_timestamp = ?
                        WHERE user_name = ?
                    """,
                    (new_password, token, timestamp, user_name))
                c.commit()
            return {"username":user_name, "token":token}
        else:
            raise PermissionError("Invalid credentials")

    def delete_user(self, user_name: str, password: str) -> None:
        if self.__verifiy_user_password(password, self.__fetch_password(user_name)):
            with sqlite3.connect(USERS_DB) as c:
                q = c.execute(    
                    """
                        DELETE
                        FROM users
                        WHERE user_name = ?
                    """,
                    (user_name,))
                c.commit()
        else:
            raise PermissionError("Invalid credentials")

    def __hash_user_password(self, password: str) -> str:
        return self.password_hash.hash(password)

    def __verifiy_user_password(self, password: str, stored_password: str) -> bool | None:
        return self.password_hash.verify(password, stored_password)

    def __user_exists(self, user_name) -> bool | None:
        with sqlite3.connect(USERS_DB) as c:
            q = c.execute(    
                """
                    SELECT user_name
                    FROM users
                    WHERE user_name = ?
                """,
                (user_name,)).fetchall()

            return len(q) > 0
    
    def __fetch_password(self, user_name: str) -> str | None:
        with sqlite3.connect(USERS_DB) as c:
            q = c.execute(    
                """
                    SELECT
                    stored_password
                    FROM users
                    WHERE user_name = ?
                """,
                (user_name,)).fetchone()
            c.commit()

            return q[0]

    def __fetch_token(self, user_name: str) -> tuple | None:
        with sqlite3.connect(USERS_DB) as c:
            q = c.execute(    
                """
                    SELECT
                    current_token, token_timestamp
                    FROM users
                    WHERE user_name = ?
                """,
                (user_name,)).fetchone()
            c.commit()

            return q

    def update_token(self, user_name: str):
        timestamp = time.time()
        token = self.__generate_jwt_token({'bearer':user_name, 'start_time':timestamp})

        with sqlite3.connect(USERS_DB) as c:
            q = c.execute(    
                """
                    UPDATE
                    users
                    SET
                    current_token = ?, token_timestamp = ?
                    WHERE user_name = ?
                """,
                (token, timestamp, user_name))
            c.commit()

            return token

    def __generate_jwt_token(self, encoding_info: dict) -> dict:
        return jwt.encode(encoding_info, SECRET_KEY, algorithm=self.algorithm)

    def authenticate_user(self, user_name: str, password: str) -> dict | None:
        if self.__user_exists(user_name):
            stored_password = self.__fetch_password(user_name)
            if self.__verifiy_user_password(password, stored_password):
                return {"username":user_name, "token":self.return_user_token(user_name)}

        return None

    def return_user_token(self, user_name: str) -> str | None:
        return self.update_token(user_name)

    def verify_user_token(self, token: str) -> str | None:
        payload = jwt.decode(token, SECRET_KEY, self.algorithm)
      
        user_name = payload.get('bearer')
        timestamp = payload.get('start_time')
        stored_token = self.__fetch_token(user_name)[0]
        if hmac.compare_digest(token, stored_token) and time.time() - float(timestamp) < self.session_timeout:
            return user_name

        return None

    def get_user_id(self, username: str) -> str | None:
        with sqlite3.connect(USERS_DB) as c:
            q = c.execute(    
                """
                    SELECT user_id
                    FROM
                    users
                    WHERE user_name = ?
                """,
                (username,)).fetchone()
            c.commit()

            return q[0]
