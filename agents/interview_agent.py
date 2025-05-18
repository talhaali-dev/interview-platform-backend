from typing import Dict, Any, List

from .base_agent import BaseAgent
from models.schemas import DifficultyLevel, InterviewRequest, QuestionResponse

class JobInterviewAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert technical interviewer with deep knowledge across various technical domains.
Your task is to generate relevant interview questions based on the provided job description, resume, and required skills.
Each question should be tailored to assess specific skills and match the requested difficulty level."""

    async def generate(self, request: InterviewRequest) -> QuestionResponse:
        prompt = self._build_prompt(
            difficulty=request.difficulty,
            job_description=request.job_description,
            resume=request.resume,
            num_questions=request.num_questions,
            skills=request.skills
        )
        
        output_schema = {
            "questions": [
                {
                    "text": "string",
                    "difficulty": "string (exactly one of: beginner, intermediate, advanced, expert)",
                    "skill_tag": "string (exactly one of: technical, system_design, problem_solving, coding)",
                    "explanation": "string"
                }
            ],
            "metadata": {
                "role_summary": "string",
                "skill_distribution": {
                    "technical": "number",
                    "system_design": "number",
                    "problem_solving": "number",
                    "coding": "number"
                }
            }
        }
        
        response = await self.llm.run_structured(
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.8
        )
        
        return QuestionResponse(**response)

    def _build_prompt(
        self,
        difficulty: DifficultyLevel,
        job_description: str,
        resume: str,
        num_questions: int,
        skills: List[str]
    ) -> str:
        return f"""Based on the following information, generate {num_questions} interview questions:

Job Description:
{job_description}

Candidate Resume:
{resume}

Required Skills:
{', '.join(skills)}

Difficulty Level: {difficulty.value}

Requirements:
1. Generate exactly {num_questions} questions
2. Each question should focus on one or more of the required skills
3. Questions should be at {difficulty.value} difficulty level
4. Include a brief explanation for each question
5. Ensure questions are varied and cover different aspects of the required skills
6. Include a role summary and skill distribution statistics in the metadata
7. Each question must use one of these skill tags: technical, system_design, problem_solving, or coding
8. The metadata must include skill_distribution showing the count of questions per skill tag

Please provide the response in the specified JSON format."""
