from fastapi import APIRouter, HTTPException
from .translater import Translater
from .translater_schema import Translater_text_Response, Translater_text_Request
import asyncio

router = APIRouter()
translater = Translater()

@router.post("/translater", response_model=Translater_text_Response)
async def translate_text(request: Translater_text_Request):
    try:
        import time

        start = time.time()
        translation_result =  translater.get_response(request.model_dump())
        

        response_data = {
            **request.model_dump(),
            **translation_result.model_dump()
        }
        import time
        print("LLM time:", time.time() - start)

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
