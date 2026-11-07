from fastapi import APIRouter, HTTPException
from .translater import Translater, TranslationServiceUnavailableError
from .translater_schema import Translater_text_Response, Translater_text_Request
import asyncio
import time

router = APIRouter()
translater = Translater()

@router.post("/translater", response_model=Translater_text_Response)
async def translate_text(request: Translater_text_Request):
    try:
        start = time.time()
        translation_result = translater.get_response(request.model_dump())

        response_data = {
            **request.model_dump(),
            **translation_result.model_dump()
        }
        print("LLM time:", time.time() - start)

        return response_data

    except TranslationServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
