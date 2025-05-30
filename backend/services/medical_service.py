import urllib.request
import urllib.error
import json
import asyncio
from typing import List, Tuple, Optional
from fastapi import HTTPException

from backend.models.chat import MedicalServicesApiResponse, MedicalServiceItem

def _fetch_medical_services_sync() -> List[Tuple[str, str]]:
    """
    Synchronous function to fetch medical services from the external API.
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing (id, name) for each medical service
    
    Raises:
        HTTPException: If the API call fails or returns invalid data
    """
    api_url = "https://medbe.actvn.live/api/v1/medical-service"
    
    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to fetch medical services. API returned status: {response.status}"
                )
            
            data = json.loads(response.read().decode('utf-8'))
            
            # Parse the response using Pydantic model
            api_response = MedicalServicesApiResponse(**data)
            
            # Extract id and name from each service item
            services = []
            for item in api_response.data.items:
                services.append((item.id, item.name))
            
            return services
        
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Network error while fetching medical services: {str(e)}"
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON response from medical services API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while fetching medical services: {str(e)}"
        )

async def fetch_medical_services() -> List[Tuple[str, str]]:
    """
    Async wrapper for fetching medical services from the external API.
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing (id, name) for each medical service
    
    Raises:
        HTTPException: If the API call fails or returns invalid data
    """
    return await asyncio.to_thread(_fetch_medical_services_sync) 