from datetime import datetime
from typing import List, Optional, Annotated, Any, ClassVar
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        """
        Return a schema that validates ObjectId strings
        and converts them to PyObjectId instances
        """
        from pydantic_core import core_schema
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ]),
        ])

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Return a schema dict for the ObjectId type"""
        return {"type": "string"}

class Message(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    content: str
    is_user: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class Conversation(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

# Request and response models for API
class MessageCreate(BaseModel):
    content: str
    is_user: bool = True

class MessageQuickChat(BaseModel):
    content: str

class ConversationCreate(BaseModel):
    user_id: str
    title: str = "New Conversation"
    initial_message: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: str

class MessageResponse(BaseModel):
    id: str = Field(alias="_id")
    content: str
    is_user: bool
    created_at: datetime

    # recommended_service_ids: Optional[List[str]]
    
    model_config = {
        "populate_by_name": True
    }

class QuickMessageResponse(BaseModel):
    content: str
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }

class RecommendResponse(BaseModel):
    content: List[str]
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }

class ConversationResponse(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    
    model_config = {
        "populate_by_name": True
    }

class MedicalServicePydantic(BaseModel):
    id: str = Field(description="Unique identifier for the medical service")
    name: str = Field(description="Name of the medical service")
    description: str = Field(description="Detailed description of the medical service")
    relatedService: str = Field(description="Related service identifier")
    symptom: str = Field(description="Symptom associated with the medical service")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "SRV001",
                    "name": "General Check-up",
                    "description": "A comprehensive general health check-up."
                }
            ]
        }
    }

class ServiceRecommendationRequest(BaseModel):
    # conversation_id will be a path parameter, so the body can be empty or used for future params
    # For now, let's make it expect an empty body or specific future params.
    # If we strictly want no body, we wouldn't define this model for the request body.
    pass # No fields expected in the body for now

class ServiceRecommendationResponseData(BaseModel):
    recommended_service_ids: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow) 

class MedicalServiceItem(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str]
    relatedService: Optional[str]
    symptom: Optional[str]

    
    model_config = {
        "populate_by_name": True
    }

class MedicalServicesApiData(BaseModel):
    items: List[MedicalServiceItem]

class MedicalServicesApiResponse(BaseModel):
    data: MedicalServicesApiData

class MedicalServiceResponse(BaseModel):
    id: str
    name: str

class MedicalServicesListResponse(BaseModel):
    services: List[MedicalServiceResponse]
    total_count: int 