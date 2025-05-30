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

    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'
    }

    try:
        request = urllib.request.Request(api_url, headers=hdr)
        with urllib.request.urlopen(request, timeout=30) as response:
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
        print(e)
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