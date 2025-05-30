from fastapi import APIRouter, HTTPException, Depends, Query, Body, status
from typing import List, Optional, Any, Tuple
from bson import ObjectId
from datetime import datetime

from backend.db.mongodb import mongodb
from backend.models.chat import (
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageResponse, QuickMessageResponse, MessageQuickChat, RecommendResponse,
    ServiceRecommendationRequest, ServiceRecommendationResponseData,
    MedicalServicesListResponse, MedicalServiceResponse
)

from backend.models.responses import BaseResponse
from agent.rag import rag, dedicated_service_recommend_chain, recommend
from backend.services.medical_service import fetch_medical_services

router = APIRouter()

# Helper function to check if ObjectId is valid
def validate_object_id(id: str):
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {id}")
    return ObjectId(id)


@router.get("/conversations/all", response_model=BaseResponse[List[ConversationResponse]])
async def get_all_conversations(
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100)
):
    cursor = mongodb.db.conversations.find(
        {}
    ).sort("updated_at", -1).skip(skip).limit(limit)

    conversations = []
    async for conv in cursor:
        # Count messages for this conversation
        response_data = ConversationResponse(
            _id=str(conv["_id"]),
            user_id=str(conv["user_id"]),
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
        )
        conversations.append(response_data)

    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Conversations retrieved successfully",
        data=conversations
    )

@router.get("/conversations", response_model=BaseResponse[List[ConversationResponse]])
async def get_conversations_of_user(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """Get all conversations for a user"""
    user_id_obj = validate_object_id(user_id)
    
    cursor = mongodb.db.conversations.find(
        {"user_id": user_id_obj}
    ).sort("updated_at", -1).skip(skip).limit(limit)
    
    conversations = []
    async for conv in cursor:
        # Count messages for this conversation
        response_data = ConversationResponse(
            _id = str(conv["_id"]),
            user_id = str(conv["user_id"]),
            title = conv["title"],
            created_at = conv["created_at"],
            updated_at = conv["updated_at"],
        )
        conversations.append(response_data)
    
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Conversations retrieved successfully",
        data=conversations
    )

@router.post("/conversations", response_model=BaseResponse[ConversationResponse])
async def create_conversation(
    conversation: ConversationCreate,
):
    """Create a new conversation"""
    user_id_obj = validate_object_id(conversation.user_id)
    
    now = datetime.utcnow()
    new_conversation = {
        "user_id": user_id_obj,
        "title": conversation.title,
        "created_at": now,
        "updated_at": now
    }
    
    result = await mongodb.db.conversations.insert_one(new_conversation)
    print("conv id: ", result)

    conversation_id = result.inserted_id
    
    created_conversation = await mongodb.db.conversations.find_one({"_id": conversation_id})

    print("Created conv")
    print(created_conversation)

    response_data = ConversationResponse(
        _id=str(created_conversation["_id"]),
        user_id=str(created_conversation["user_id"]),
        title= created_conversation["title"],
        created_at = created_conversation["created_at"],
        updated_at = created_conversation["updated_at"],
    )
    
    return BaseResponse(
        statusCode=status.HTTP_201_CREATED,
        message="Conversation created successfully",
        data=response_data
    )

@router.put("/conversations/{conversation_id}", response_model=BaseResponse[ConversationResponse])
async def update_conversation(
    conversation_id: str,
    conversation: ConversationUpdate,
):
    """Update a conversation's title"""
    conv_id = validate_object_id(conversation_id)

    result = await mongodb.db.conversations.update_one(
        {"_id": conv_id},
        {"$set": {"title": conversation.title, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    updated_conversation = await mongodb.db.conversations.find_one({"_id": conv_id})

    response_data = ConversationResponse(
        _id=str(updated_conversation["_id"]),
        user_id=str(updated_conversation["user_id"]),
        title=updated_conversation["title"],
        created_at=updated_conversation["created_at"],
        updated_at=updated_conversation["updated_at"],
    )
    
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Conversation updated successfully",
        data=response_data
    )

@router.delete("/conversations/{conversation_id}", response_model=BaseResponse)
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    conv_id = validate_object_id(conversation_id)

    # First check if conversation exists
    conversation = await mongodb.db.conversations.find_one(
        {"_id": conv_id}
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete all messages in the conversation
    await mongodb.db.messages.delete_many({"conversation_id": conv_id})
    
    # Delete the conversation
    await mongodb.db.conversations.delete_one({"_id": conv_id})
    
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Conversation deleted successfully",
        data=None
    )

@router.get("/messages/{conversation_id}", response_model=BaseResponse[List[MessageResponse]])
async def get_messages_of_conversation(
    conversation_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """Get messages for a specific conversation"""
    conv_id = validate_object_id(conversation_id)

    # Check if conversation exists and belongs to the user
    conversation = await mongodb.db.conversations.find_one(
        {"_id": conv_id}
    )

    print("Conv:")
    print(conversation)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    cursor = mongodb.db.messages.find(
        {"conversation_id": conv_id}
    ).sort("created_at", 1).skip(skip).limit(limit)
    
    messages = []
    async for msg in cursor:
        response_msg = MessageResponse(
            _id = str(msg["_id"]),
            content = msg["content"],
            is_user = msg["is_user"],
            created_at = msg["created_at"],
        )
        messages.append(response_msg)

    print("all messages")
    print(messages)
    
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Messages retrieved successfully",
        data=messages
    )

@router.post("/{conversation_id}/messages", response_model=BaseResponse[MessageResponse])
async def query_ai(
    conversation_id: str,
    message: MessageCreate,
):
    """Add a new message to a conversation"""
    conv_id = validate_object_id(conversation_id)

    # Check if conversation exists and belongs to the user
    conversation = await mongodb.db.conversations.find_one(
        {"_id": conv_id}
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    now = datetime.utcnow()

    # Create the user message
    new_message = {
        "conversation_id": conv_id,
        "content": message.content,
        "is_user": message.is_user,
        "created_at": now
    }

    await mongodb.db.messages.insert_one(new_message)

    # Update the conversation's updated_at timestamp
    await mongodb.db.conversations.update_one(
        {"_id": conv_id},
        {"$set": {"updated_at": now}}
    )

    ## Get all messages history
    cursor = mongodb.db.messages.find(
        {"conversation_id": conv_id}
    ).sort("created_at", 1)

    messages = []
    async for msg in cursor:
        messages.append(msg)

    model_chat_history = []
    for msg in messages:
        if msg["is_user"]:
            ai_msg = "[user]" + msg["content"]
        else:
            ai_msg = "[bot]" + msg["content"]

        model_chat_history.append(ai_msg)

    response = rag.invoke({
        "input": message.content,
        "chat_history": model_chat_history[:-1],  # Exclude the current message
    })

    answer = response["answer"]

    now = datetime.utcnow()

    # Create the bot message
    new_message = {
        "conversation_id": conv_id,
        "content": answer,
        "is_user": False,
        "created_at": now
    }

    result = await mongodb.db.messages.insert_one(new_message)

    # Update the conversation's updated_at timestamp
    await mongodb.db.conversations.update_one(
        {"_id": conv_id},
        {"$set": {"updated_at": now}}
    )

    created_message = await mongodb.db.messages.find_one({"_id": result.inserted_id})

    response_data = MessageResponse(
        _id=str(created_message["_id"]),
        content=created_message["content"],
        is_user=created_message["is_user"],
        created_at=created_message["created_at"],
    )

    return BaseResponse(
        statusCode=status.HTTP_201_CREATED,
        message="Message created successfully",
        data=response_data
    )


@router.post("/quick-messages", response_model=BaseResponse[QuickMessageResponse])
async def quick_chat(
    message: MessageQuickChat,
):
    model_chat_history = []
    ai_msg = "[user]" + message.content

    model_chat_history.append(ai_msg)

    response = rag.invoke({
        "input": message.content,
        "chat_history": model_chat_history[:-1],  # Exclude the current message
    })

    answer = response["answer"]

    now = datetime.utcnow()


    response_data = QuickMessageResponse(
        content=answer,
        created_at=now,
    )

    return BaseResponse(
        statusCode=status.HTTP_201_CREATED,
        message="Message created successfully",
        data=response_data
    )

@router.post("/recommend", response_model=BaseResponse[RecommendResponse])
async def quick_chat(
    message: MessageQuickChat,
):
    model_chat_history = []
    ai_msg = "[user]" + message.content

    model_chat_history.append(ai_msg)

    response = recommend.invoke({
        "input": message.content,
        "chat_history": model_chat_history[:-1],  # Exclude the current message
    })

    answer = response["answer"]

    now = datetime.utcnow()

    ## Split answer string by "|"
    recommendations = [rec.strip() for rec in answer.split("|") if rec.strip()]

    if not recommendations:
        raise HTTPException(status_code=400, detail="No recommendations found in the response")

    response_data = RecommendResponse(
        content=recommendations,
        created_at=now,
    )

    return BaseResponse(
        statusCode=status.HTTP_201_CREATED,
        message="Message created successfully",
        data=response_data
    )

@router.post("/conversations/{conversation_id}/recommend", response_model=BaseResponse[RecommendResponse])
async def recommend_from_conversation(
    conversation_id: str,
):
    """Generate recommendations based on conversation history."""
    conv_id = validate_object_id(conversation_id)

    # Check if conversation exists
    conversation = await mongodb.db.conversations.find_one({"_id": conv_id})
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Get all messages for the conversation
    message_cursor = mongodb.db.messages.find(
        {"conversation_id": conv_id}
    ).sort("created_at", 1)
    
    messages = []
    async for msg in message_cursor:
        messages.append(msg)

    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Conversation has no messages to base recommendations on.")

    # Build chat history in the format expected by the recommend chain
    model_chat_history = []
    for msg in messages:
        if msg["is_user"]:
            ai_msg = "[user]" + msg["content"]
        else:
            ai_msg = "[bot]" + msg["content"]
        model_chat_history.append(ai_msg)

    # Find the last user message as input
    last_user_message = None
    for msg in reversed(messages):
        if msg["is_user"]:
            last_user_message = msg["content"]
            break
    
    if not last_user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No user messages found in conversation.")

    # Invoke the recommend chain
    try:
        response = recommend.invoke({
            "input": last_user_message,
            "chat_history": model_chat_history[:-1] if len(model_chat_history) > 0 else [],
        })

        answer = response["answer"]
    except Exception as e:
        print(f"Error invoking recommend chain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get recommendations: {str(e)}")

    now = datetime.utcnow()

    # Split answer string by "|"
    recommendations = [rec.strip() for rec in answer.split("|") if rec.strip()]

    if not recommendations:
        raise HTTPException(status_code=400, detail="No recommendations found in the response")

    response_data = RecommendResponse(
        content=recommendations,
        created_at=now,
    )

    return BaseResponse(
        statusCode=status.HTTP_201_CREATED,
        message="Recommendations created successfully",
        data=response_data
    )

@router.post("/conversations/{conversation_id}/service-recommendations", response_model=BaseResponse[ServiceRecommendationResponseData])
async def recommend_services(
    conversation_id: str,
    # request_body: ServiceRecommendationRequest = Body(...) # Use if body params are added
):
    """Recommend medical services based on conversation history."""
    conv_id = validate_object_id(conversation_id)

    # Check if conversation exists
    conversation = await mongodb.db.conversations.find_one({"_id": conv_id})
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Get all messages for the conversation to build history and find the last user message
    message_cursor = mongodb.db.messages.find(
        {"conversation_id": conv_id}
    ).sort("created_at", 1)
    
    chat_history_for_chain = []
    all_messages_in_db = [] # To store full message objects from DB
    async for msg in message_cursor:
        all_messages_in_db.append(msg)
        # Langchain expects history as list of (human_input, ai_response) tuples or BaseMessages
        # Here, we adapt the stored format [user] content, [bot] content
        # For the service_recommend chain, we are passing it as (role, content) tuples
        role = "user" if msg["is_user"] else "bot"
        chat_history_for_chain.append((role, msg["content"]))

    if not all_messages_in_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Conversation has no messages to base recommendations on.")

    # The service_recommend chain expects the current "input" and "chat_history"
    # We'll use the content of the last message as the current "input".
    # The chat_history will be all messages *before* the last one.
    if not chat_history_for_chain:
         # Should not happen if all_messages_in_db is not empty, but good for safety
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot process empty chat history for recommendations.")

    last_message_content = chat_history_for_chain[-1][1] # Content of the last message
    history_up_to_last = chat_history_for_chain[:-1] # All messages except the last one

    # Invoke the service recommendation chain
    try:
        # The input to the dedicated chain is {"input": str, "chat_history": List[Tuple[str,str]]}
        recommendation_output = dedicated_service_recommend_chain.invoke({
            "input": last_message_content, 
            "chat_history": history_up_to_last
        })
        # The chain now returns a ServiceRecommendationOutput Pydantic model directly
        recommended_ids = recommendation_output.recommended_service_ids

    except Exception as e:
        # Log the exception for debugging
        print(f"Error invoking service_recommend chain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get service recommendations: {str(e)}")

    response_data = ServiceRecommendationResponseData(
        recommended_service_ids=recommended_ids
    )

    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="Service recommendations retrieved successfully",
        data=response_data
    )

@router.get("/medical-services", response_model=BaseResponse[MedicalServicesListResponse])
async def get_medical_services():
    """
    Fetch all available medical services from the external API.
    
    Returns:
        BaseResponse containing list of medical services with id and name
    """
    try:
        services_data = await fetch_medical_services()
        
        # Convert tuples to MedicalServiceResponse objects
        services = [
            MedicalServiceResponse(id=service_id, name=service_name) 
            for service_id, service_name in services_data
        ]
        
        response_data = MedicalServicesListResponse(
            services=services,
            total_count=len(services)
        )
        
        return BaseResponse(
            statusCode=status.HTTP_200_OK,
            message="Medical services retrieved successfully",
            data=response_data
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions from the service function
        raise e
    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve medical services: {str(e)}"
        )
