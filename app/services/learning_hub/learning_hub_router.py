from fastapi import APIRouter, HTTPException
from .learning_hub_schema import Learning_hub_Request, LearningHub_Response
from .learning_hub import LearningHub
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
learning_hub_generator = LearningHub()

@router.post("/learning_hub", response_model=LearningHub_Response)
async def learning_hub(request: Learning_hub_Request):
    try:
        user_prompt_text = request.user_prompt
        lesson = learning_hub_generator.get_response(user_prompt_text)
        return lesson
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))