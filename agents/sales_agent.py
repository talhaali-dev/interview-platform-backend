import base64
from typing import Dict, Any
from pypdf import PdfReader
from io import BytesIO

from .base_agent import BaseAgent
from models.schemas import SalesRequest, QuestionResponse

class SalesCallAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an experienced sales professional specializing in consultative selling.
Your task is to analyze the provided project scope and generate strategic discovery and objection-handling questions.
Focus on understanding the client's needs, pain points, and potential objections."""

    async def generate(self, request: SalesRequest) -> QuestionResponse:
        # Extract text from PDF if the input is base64 encoded PDF
        try:
            pdf_bytes = base64.b64decode(request.project_scope)
            scope_text = self._extract_pdf_text(pdf_bytes)
        except:
            # If not base64 or PDF, assume it's raw text
            scope_text = request.project_scope
            
        prompt = self._build_prompt(scope_text)
        
        output_schema = {
            "questions": [
                {
                    "text": "string",
                    "difficulty": "string",
                    "skill_tag": "string (one of: discovery, objection-handling, needs-analysis, value-proposition)",
                    "explanation": "string"
                }
            ],
            "metadata": {
                "scope_summary": "string",
                "key_points": ["string"],
                "potential_challenges": ["string"]
            }
        }
        
        response = await self.llm.run_structured(
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7
        )
        
        return QuestionResponse(**response)

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        pdf = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

    def _build_prompt(self, scope_text: str) -> str:
        return f"""Based on the following project scope, generate 8-12 strategic sales questions:

Project Scope:
{scope_text}

Requirements:
1. Generate a mix of discovery and objection-handling questions
2. Questions should help understand the client's:
   - Current challenges and pain points
   - Business objectives and success criteria
   - Decision-making process and stakeholders
   - Budget and timeline constraints
3. Include questions that:
   - Uncover unstated needs
   - Probe deeper into stated requirements
   - Address potential objections preemptively
4. For each question, provide:
   - The question text
   - The type of question (discovery/objection-handling)
   - A brief explanation of why to ask this question
5. Include in metadata:
   - A summary of the project scope
   - Key points extracted
   - Potential challenges identified

Please provide the response in the specified JSON format."""
