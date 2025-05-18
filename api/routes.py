from fastapi import APIRouter, Depends, HTTPException
from typing import Any

from core.config import get_settings, Settings
from core.llm_client import LLMClient
from agents.interview_agent import JobInterviewAgent
from agents.sales_agent import SalesCallAgent
from agents.english_agent import EnglishPrepAgent
from models.schemas import (
    InterviewRequest,
    SalesRequest,
    EnglishPrepRequest,
    QuestionResponse
)

router = APIRouter(prefix="/api/v1")

async def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return LLMClient(api_key=settings.GOOGLE_API_KEY)

@router.post("/interview", response_model=QuestionResponse)
async def generate_interview_questions(
    request: InterviewRequest,
    llm_client: LLMClient = Depends(get_llm_client)
) -> Any:
    """Generate technical interview questions based on job description and resume."""
    try:
        agent = JobInterviewAgent(llm_client)
        response = await agent.generate(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate interview questions: {str(e)}"
        )

@router.post("/sales", response_model=QuestionResponse)
async def generate_sales_questions(
    request: SalesRequest,
    llm_client: LLMClient = Depends(get_llm_client)
) -> Any:
    """Generate sales discovery and objection handling questions from project scope."""
    try:
        agent = SalesCallAgent(llm_client)
        response = await agent.generate(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sales questions: {str(e)}"
        )

@router.post("/english", response_model=QuestionResponse)
async def generate_english_questions(
    request: EnglishPrepRequest,
    llm_client: LLMClient = Depends(get_llm_client)
) -> Any:
    """Generate English language practice questions."""
    try:
        agent = EnglishPrepAgent(llm_client)
        response = await agent.generate(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate English questions: {str(e)}"
        )
