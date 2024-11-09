"""LLM service implementations."""

import os
from ..core.cache import Dependencies

class LLMService:
    """Manages LLM operations."""
    
    def __init__(self, 
                 model_name,
                 model_type,
                 temperature=0.1,
                 max_tokens=5000):
        self.model_name = model_name
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._deps = Dependencies()
        
    def get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            ChatOpenAI, ChatAnthropic = self._deps.get_llm_deps()
            
            if self.model_type == 'OpenAI':
                self._llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    tags=[self.model_name]
                )
            elif self.model_type == 'Anthropic':
                self._llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    tags=[self.model_name]
                )
            elif self.model_type == 'Hugging Face':
                hf_endpoint = 'https://api-inference.huggingface.co/v1'
                self._llm = ChatOpenAI(
                    base_url=hf_endpoint,
                    model=self.model_name,
                    api_key=os.getenv('HUGGINGFACEHUB_API_KEY'),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tags=[self.model_name]
                )
            # TODO Test local llm with lm studio
            elif self.model_type == 'LM Studio (local)':
                self._llm = ChatOpenAI(
                    base_url=self.model_name,  # For local LLMs
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    tags=[self.model_name]
                )
            else:
                raise ValueError(f"Unsupported LLM type: {self.model_type}")
        return self._llm