"""
Database connection and session management
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.postgres_url = self._build_postgres_url()
        self.engine = create_engine(
            self.postgres_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=os.getenv('DB_ECHO', 'False').lower() == 'true'
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def _build_postgres_url(self) -> str:
        """Build PostgreSQL connection URL"""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'finriskai')
        user = os.getenv('POSTGRES_USER', 'finrisk_user')
        password = os.getenv('POSTGRES_PASSWORD', 'secure_password123')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """Get database session for dependency injection"""
        return self.SessionLocal()

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Database dependency for FastAPI"""
    db = db_manager.get_session_direct()
    try:
        yield db
    finally:
        db.close()