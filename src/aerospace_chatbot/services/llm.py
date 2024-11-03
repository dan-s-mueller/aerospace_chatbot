"""LLM service implementations."""

from ..core.cache import Dependencies

class LLMService:
    """Manages LLM operations."""
    
    def __init__(self, 
                 model_name,
                 model_type,
                 api_key,
                 temperature=0.1,
                 max_tokens=1000):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
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
                    openai_api_key=self.api_key
                )
            elif self.model_type == 'Anthropic':
                self._llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.api_key
                )
            elif self.model_type == 'LM Studio (local)':
                self._llm = ChatOpenAI(
                    base_url=self.model_name,  # For local LLMs
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=self.api_key
                )
            else:
                raise ValueError(f"Unsupported LLM type: {self.model_type}")
        return self._llm