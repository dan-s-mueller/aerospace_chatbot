"""LLM service implementations."""

import os
from ..core.cache import Dependencies, cache_resource

class LLMService:
    """Manages LLM operations."""
    
    def __init__(self, 
                 model_service,
                 model,
                 temperature=0.1,
                 max_tokens=5000):
        self.model_service = model_service
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        
    def get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            ChatOpenAI, ChatAnthropic = Dependencies.LLM.get_models()
            
            if self.model_service == 'OpenAI':
                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    tags=[self.model]
                )
            elif self.model_service == 'Anthropic':
                self._llm = ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    tags=[self.model]
                )
            elif self.model_service == 'Hugging Face':
                hf_endpoint = 'https://api-inference.huggingface.co/v1'
                self._llm = ChatOpenAI(
                    base_url=hf_endpoint,
                    model=self.model,
                    api_key=os.getenv('HUGGINGFACEHUB_API_KEY'),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tags=[self.model]
                )
            # TODO Test local llm with lm studio
            elif self.model_service == 'LM Studio (local)':
                self._llm = ChatOpenAI(
                    base_url=self.model,  # For local LLMs
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    tags=[self.model]
                )
            else:
                raise ValueError(f"Unsupported LLM type: {self.model_service}")
        return self._llm
