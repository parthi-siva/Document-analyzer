from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Optional

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseChatHistory:
    def __init__(self, database_url: str = "sqlite:///chat_history.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def add_message(self, session_id: str, role: str, content: str):
        db = self.SessionLocal()
        try:
            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            db.commit()
        finally:
            db.close()
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        db = self.SessionLocal()
        try:
            messages = db.query(ChatMessage)\
                        .filter(ChatMessage.session_id == session_id)\
                        .order_by(ChatMessage.timestamp.desc())\
                        .limit(limit)\
                        .all()
            
            # Convert to chat format
            history = []
            for msg in reversed(messages):  # Reverse to get chronological order
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            return history
        finally:
            db.close()

