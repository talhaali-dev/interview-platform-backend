from abc import ABC, abstractmethod
from typing import Dict, Any

from core.llm_client import LLMClient

class BaseAgent(ABC):
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    @abstractmethod
    async def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a response based on the input parameters.
        Must be implemented by concrete agent classes.
        """
        pass
    
    def _build_prompt(self, **kwargs) -> str:
        """
        Helper method to build prompts. Can be overridden by concrete classes.
        """
        raise NotImplementedError("Prompt building not implemented for this agent")
