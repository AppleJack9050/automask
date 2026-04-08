from uuid import uuid4
import sqlite3
import os
import re

FILES_DB = os.getenv('FILES_DB')

class FilesTable:
    def __init__(self):
        self.init_db()

    def init_db(self):
        db_dir = os.path.dirname(FILES_DB)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(FILES_DB) as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                file_name TEXT,
                stored_file_name TEXT
            )
            """)
            c.commit()

    def insert_name(self, file_name: str, user: str, ) -> str | None:
        count = self.__count_file_duplicates(file_name, user)

        file_extension = ""
        if file_name.endswith(".tar.gz"):
            file_extension = ".tar.gz"
            file_name = file_name[0:-6]

        else:
            file_extension = "." + file_name.split('.')[-1]

        if count > 0:
            file_name = f"{file_name.removesuffix(file_extension)} ({count}){file_extension}"

        with sqlite3.connect(FILES_DB) as c:
            stored_file_name = str(uuid4()) + file_extension
            c.execute(
                """
                    INSERT INTO files (id, user_id, file_name, stored_file_name)
                    VALUES (?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    user,
                    file_name,
                    stored_file_name
                ))
            c.commit()
            return stored_file_name

    def get_stored_name(self, file_name: str, user: str) -> str | None:
        with sqlite3.connect(FILES_DB) as c:
            q = c.execute(    
                """
                    SELECT
                    stored_file_name
                    FROM files WHERE user_id = ? AND file_name = ?
                """,
                (user, file_name)).fetchone()
            c.commit()
        if not q:
            return ""
        return q[0]
        
    def get_actual_name(self, stored_file_name: str, user: str) -> str | None:
        with sqlite3.connect(FILES_DB) as c:
            q = c.execute(    
                """
                    SELECT
                    file_name
                    FROM files WHERE user_id = ? AND stored_file_name LIKE ?
                """,
                (user, f"%{stored_file_name}%")).fetchone()
            c.commit()

        if not q:
            return None
        return q[0]

    def delete_name(self, file_name: str, user: str):
        with sqlite3.connect(FILES_DB) as c:
            c.execute(    
                """
                    DELETE 
                    FROM files
                    WHERE user_id = ?. file_name = ?
                """,
                (user, file_name)).fetchone()
            c.commit()

    def __count_file_duplicates(self, file_name: str, user_id: str) -> int:
        name, _, ext = file_name.rpartition('.')
        with sqlite3.connect(FILES_DB) as c:
            q = c.execute(    
                """
                    SELECT file_name
                    FROM files
                    WHERE user_id = ? AND file_name LIKE ?
                """,
                (user_id, f"%{file_name}%")).fetchall()
            c.commit()
        stem = re.escape(name)
        exact_pattern = re.compile(rf"^{stem}( \(\d+\))?.{re.escape(ext)}$")
        return sum(1 for (fname,) in q if exact_pattern.match(fname))

    def delete_user_files(self, user_id: str):
        with sqlite3.connect(FILES_DB) as c:
            q = c.execute(    
                """
                    DELETE
                    FROM files
                    WHERE user_id = ?
                """,
                (user_id,)).fetchall()
            c.commit()

    def share_file(self, file_recipient_id: str, file_name: str, stored_file_name: str):
        with sqlite3.connect(FILES_DB) as c:
            c.execute(
                """
                    INSERT INTO files (id, user_id, file_name, stored_file_name)
                    VALUES (?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    file_recipient_id,
                    file_name,
                    stored_file_name
                ))
            c.commit()
