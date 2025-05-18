from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from models.schemas import DifficultyLevel, EnglishPrepRequest, QuestionResponse

class EnglishPrepAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert English language instructor with experience in teaching at all levels.
Your task is to generate English language practice questions that test grammar, vocabulary, and comprehension skills.
Each question should include a clear explanation and correct answer."""

    async def generate(self, request: EnglishPrepRequest) -> QuestionResponse:
        prompt = self._build_prompt(
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            topic=request.topic
        )
        
        output_schema = {
            "questions": [
                {
                    "text": "string",
                    "difficulty": "string (one of: beginner, intermediate, advanced, expert)",
                    "skill_tag": "string (one of: grammar, vocabulary, comprehension, speaking, writing)",
                    "explanation": "string",
                    "answer": "string"
                }
            ],
            "metadata": {
                "topic_focus": "string",
                "skill_distribution": "dict[str, int]",
                "learning_objectives": ["string"]
            }
        }
        
        response = await self.llm.run_structured(
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7
        )
        
        return QuestionResponse(**response)

    def _build_prompt(
        self,
        difficulty: DifficultyLevel,
        num_questions: int,
        topic: Optional[str] = None
    ) -> str:
        topic_str = f"\nFocus Topic: {topic}" if topic else "\nGenerate questions covering various topics."
        
        return f"""Generate {num_questions} English language practice questions at {difficulty.value} level.{topic_str}

Requirements:
1. Generate exactly {num_questions} questions
2. Questions should be at {difficulty.value} difficulty level
3. Include a mix of:
   - Grammar exercises
   - Vocabulary usage
   - Reading comprehension
   - Speaking/pronunciation (if applicable)
   - Writing prompts (if applicable)
4. For each question, provide:
   - The question text
   - The correct answer
   - A detailed explanation of the answer
   - The specific skill being tested
5. Include in metadata:
   - Main topic focus
   - Distribution of skills tested
   - Key learning objectives

Please provide the response in the specified JSON format."""
