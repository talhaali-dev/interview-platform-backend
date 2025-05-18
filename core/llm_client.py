import json
import google.generativeai as genai
from typing import Any, Dict, Optional
from google.generativeai.types import GenerateContentResponse
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logger.debug("Initialized Gemini model with config: %s", generation_config)
    
    async def run(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Run the LLM with the given prompt and parameters
        """
        try:
            # Combine system prompt and user prompt if both are provided
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            logger.debug("Sending prompt to Gemini: %s", full_prompt)
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    **kwargs
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
                
            logger.debug("Received response from Gemini: %s", response.text)
            return response.text
            
        except Exception as e:
            logger.error("Error in run method: %s", str(e), exc_info=True)
            raise

    async def run_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the LLM and ensure output matches the provided schema
        """
        try:
            # Create a prompt that enforces JSON output
            json_prompt = f"""
            You are a structured data generator. Your response must be valid JSON that matches the following schema:
            {json.dumps(output_schema, indent=2)}

            Important rules:
            1. Response MUST be valid JSON
            2. All fields in the schema must be present
            3. All values must match their specified types
            4. Do not include any explanatory text, ONLY the JSON object
            5. NO trailing commas in JSON objects or arrays
            6. NO comments in the JSON
            7. Use double quotes for strings, not single quotes

            Based on this request:
            {prompt}
            """
            
            logger.debug("Sending structured prompt to Gemini")
            response_text = await self.run(json_prompt, **kwargs)
            
            # Extract JSON from the response
            # First, try to find JSON block if there's any surrounding text
            try:
                # Clean the response text
                response_text = response_text.strip()
                # Remove any trailing commas before closing braces/brackets
                response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
                
                # Try to find JSON block between ```json and ```
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                else:
                    # Try to find any content between ``` and ```
                    json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                
                response_text = response_text.strip()
                response_data = json.loads(response_text)
                logger.debug("Successfully parsed JSON response: %s", response_data)
                return response_data
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON response: %s\nResponse text: %s", str(e), response_text)
                raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
                
        except Exception as e:
            logger.error("Error in run_structured method: %s", str(e), exc_info=True)
            raise
