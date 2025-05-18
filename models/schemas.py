from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class SkillTag(str, Enum):
    # Interview skills
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    PROBLEM_SOLVING = "problem_solving"
    CODING = "coding"
    
    # Sales skills
    DISCOVERY = "discovery"
    OBJECTION_HANDLING = "objection_handling"
    NEEDS_ANALYSIS = "needs_analysis"
    VALUE_PROPOSITION = "value_proposition"
    
    # English skills
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    COMPREHENSION = "comprehension"
    SPEAKING = "speaking"
    WRITING = "writing"

class Question(BaseModel):
    text: constr(min_length=10, max_length=1000)
    difficulty: DifficultyLevel
    skill_tag: SkillTag
    explanation: Optional[constr(min_length=10, max_length=1000)] = None
    answer: Optional[constr(min_length=1, max_length=500)] = None

class InterviewRequest(BaseModel):
    difficulty: DifficultyLevel
    job_description: constr(min_length=50, max_length=5000)
    resume: constr(min_length=50, max_length=5000)
    num_questions: int = Field(gt=0, le=20, description="Number of questions to generate")
    skills: List[str] = Field(min_items=1, max_items=10)

class SalesRequest(BaseModel):
    project_scope: constr(min_length=50, max_length=10000) = Field(
        description="Base64 encoded PDF or raw text of the project scope"
    )

class EnglishPrepRequest(BaseModel):
    num_questions: int = Field(gt=0, le=20, description="Number of questions to generate")
    difficulty: DifficultyLevel
    topic: Optional[constr(min_length=3, max_length=100)] = Field(
        None, description="Optional specific topic to focus on"
    )

class QuestionMetadata(BaseModel):
    # Common metadata
    skill_distribution: Dict[str, int] = Field(description="Distribution of skills covered")
    
    # Interview specific
    role_summary: Optional[str] = None
    skill_coverage: Optional[Dict[str, int]] = None
    
    # Sales specific
    scope_summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    potential_challenges: Optional[List[str]] = None
    
    # English specific
    topic_focus: Optional[str] = None
    learning_objectives: Optional[List[str]] = None

class QuestionResponse(BaseModel):
    questions: List[Question] = Field(min_items=1)
    metadata: QuestionMetadata
