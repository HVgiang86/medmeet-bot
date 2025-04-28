from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

class MongoDBSettings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "ai_chat"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = ""
    LANGCHAIN_PROJECT: str = ""
    
    class Config:
        env_file = ".env"

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None
    
    @classmethod
    async def connect_to_mongodb(cls):
        settings = MongoDBSettings()
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.MONGODB_DB_NAME]
        logger.info("Connected to MongoDB")
        
    @classmethod
    async def close_mongodb_connection(cls):
        if cls.client:
            logger.info("Closing MongoDB connection")
            cls.client.close()
            logger.info("MongoDB connection closed")

# Convenience properties to access collections
    @property
    def conversations(self):
        return self.db.conversations
        
    @property
    def messages(self):
        return self.db.messages
        
mongodb = MongoDB() 