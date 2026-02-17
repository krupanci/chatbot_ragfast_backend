import sqlite3
from datetime import datetime,timezone,date
from typing import Optional,Dict
import uuid
from .auth_utiles import get_password_hash,verify_password,User
from .error_handlers import DatabaseError,setup_logger
from .config import settings
from fastapi import FastAPI, HTTPException




logger=setup_logger("UserDB")

class UserDatabase:
    def __init__(self,db_path:str='users.db'):
        self.db_path=db_path
        self._init_database()
        self.DAILY_LIMIT = int(settings.MAX_REQUESTS_PER_MINUTE.split("/")[0])
        
    def _init_database(self):
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                                # Create index for faster lookups
                            # Upload Jobs Table (Per User)
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys = ON;")
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS upload_jobs (
                            job_id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            status TEXT NOT NULL,
                            message TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(user_id)
                        )
                            
                        """)

                    cursor.execute("""
                                CREATE INDEX IF NOT EXISTS idx_upload_user
                                ON upload_jobs(user_id)
                            """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_username ON users(username)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_email ON users(email)
                """)
                
                conn.commit()
                logger.info("User database initialized successfully")
         
        except Exception as e:
            logger.error(f"Failed to initialize user database: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
        
    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user"""
        try:
            user_id = str(uuid.uuid4())
            hashed_password = get_password_hash(password)
            created_at = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor =conn.cursor()
                cursor.execute(""" 
                               INSERT INTO users (user_id, username, email, hashed_password, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, username, email, hashed_password, created_at))
                conn.commit()
            
            logger.info(f"user created : {username}, {user_id}")
            
            return User(
                user_id=user_id,
                username=username,
                email=email,
                is_active=True,
                created_at=created_at
            )
            
        except sqlite3.IntegrityError as e:
            if "usename" in str(e):
                raise DatabaseError('Username already exsits')
            elif 'email' in str(e):
                raise DatabaseError('Email is alreay exists')
            else:
                raise DatabaseError('User creation is failed')
        except Exception as e:
            logger.error(f"failed to create user: {e}")
            raise DatabaseError(f"user creation failed: {str(e)}")
        
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, email, hashed_password, is_active, created_at
                    FROM users WHERE username = ?
                """, (username,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, email, is_active, created_at
                    FROM users WHERE user_id = ?
                """, (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
        
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user with username and password"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        if not verify_password(password, user["hashed_password"]):
            return None
        
        if not user["is_active"]:
            return None
        
        # Update last login
        self.update_last_login(user["user_id"])
        
        return user
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET last_login = ? WHERE user_id = ?
                """, (datetime.now(timezone.utc), user_id))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update last login: {e}")
            
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET is_active = 0 WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to deactivate user: {e}")
            return False
      
    def change_password(self, user_id: str, new_password: str) -> bool:
        """Change user password"""
        try:
            hashed_password = get_password_hash(new_password)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET hashed_password = ? WHERE user_id = ?
                """, (hashed_password, user_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to change password: {e}")
            return False
        
    def create_upload_job(self, user_id: str, filename: str) -> str:
        job_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO upload_jobs (job_id, user_id, filename, status, message)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, user_id, filename, "processing", "File is being processed"))
            conn.commit()

        return job_id
    
    def update_upload_job(self, job_id: str, status: str, message: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE upload_jobs
                SET status = ?, message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (status, message, job_id))
            conn.commit()

    
    
    def get_upload_job(self, user_id: str, job_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, message, filename
                FROM upload_jobs
                WHERE job_id = ? AND user_id = ?
            """, (job_id, user_id))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None
    def mark_job_deleted(self,user_id, filename):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE upload_jobs
            SET status = 'deleted'
            WHERE user_id = ? AND filename = ?
        """, (user_id, filename))

        conn.commit()
        conn.close()
        
    def mark_all_jobs_deleted(self,user_id):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE upload_jobs
            SET status = 'deleted'
            WHERE user_id = ?
        """, (user_id,))

        conn.commit()
        conn.close()
        
        

    def check_and_increment_daily_limit(self, user_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id TEXT PRIMARY KEY,
                    count INTEGER,
                    last_reset DATE
                )
            """)

            today = date.today().isoformat()

            cursor.execute(
                "SELECT count, last_reset FROM rate_limits WHERE user_id = ?",
                (user_id,)
            )

            row = cursor.fetchone()

            if row:
                count, last_reset = row
                count = int(count)

                if last_reset != today:
                    # Reset counter for new day
                    cursor.execute("""
                        UPDATE rate_limits
                        SET count = 1, last_reset = ?
                        WHERE user_id = ?
                    """, (today, user_id))
                    conn.commit()
                    return

                if count >= self.DAILY_LIMIT:
                    raise HTTPException(
                        status_code=429,
                        detail="Daily limit reached",
                        headers={"Retry-After": str(settings.RATE_LIMIT_SECONDS)}  # 24h
                    )

                cursor.execute("""
                    UPDATE rate_limits
                    SET count = count + 1
                    WHERE user_id = ?
                """, (user_id,))
            else:
                cursor.execute("""
                    INSERT INTO rate_limits (user_id, count, last_reset)
                    VALUES (?, 1, ?)
                """, (user_id, today))

            conn.commit()
        


user_db=UserDatabase()

        
        
