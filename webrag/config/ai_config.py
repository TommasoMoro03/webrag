"""
AI/LLM configuration for WebRAG components.

This module provides a flexible configuration system for AI-powered features
across the library (crawler, chunker, extractor, etc.).

Design principles:
1. Auto-detect available AI providers from environment
2. Allow explicit configuration via code or config files
3. Reusable across all pipeline components
4. No forced dependencies - gracefully degrade if AI not available
"""

import os
from enum import Enum
from typing import Optional, Callable, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI/LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class AIConfig(BaseModel):
    """
    Configuration for AI/LLM integration across WebRAG components.

    This class manages LLM settings for any component that needs AI
    (crawler, chunker, extractor, etc.).

    Examples:
        # Auto-detect from environment
        >>> config = AIConfig.from_env()

        # Explicit configuration
        >>> config = AIConfig(
        ...     provider=AIProvider.OPENAI,
        ...     model="gpt-4o-mini",
        ...     api_key="sk-..."
        ... )

        # Use with custom LLM function
        >>> config = AIConfig.from_function(my_llm_function)

        # Disable AI features
        >>> config = AIConfig(enabled=False)
    """

    enabled: bool = Field(
        default=True,
        description="Whether AI features are enabled"
    )

    provider: Optional[AIProvider] = Field(
        default=None,
        description="AI provider to use (auto-detected if None)"
    )

    model: Optional[str] = Field(
        default=None,
        description="Model name (provider-specific)"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider (not logged)"
    )

    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL (for Ollama or custom endpoints)"
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM temperature for creativity"
    )

    max_tokens: int = Field(
        default=500,
        ge=1,
        description="Maximum tokens in LLM responses"
    )

    timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds"
    )

    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )

    # Internal: custom LLM function
    _llm_function: Optional[Callable[[str], str]] = None

    class Config:
        # Allow arbitrary types for the callable
        arbitrary_types_allowed = True
        # Don't include private fields in serialization
        underscore_attrs_are_private = True

        json_schema_extra = {
            "example": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 500
            }
        }

    @classmethod
    def from_env(cls, provider: Optional[AIProvider] = None) -> "AIConfig":
        """
        Create AIConfig by auto-detecting from environment variables.

        Looks for:
        - OPENAI_API_KEY → OpenAI
        - ANTHROPIC_API_KEY → Anthropic
        - OLLAMA_BASE_URL → Ollama

        Args:
            provider: Force specific provider (optional)

        Returns:
            AIConfig instance

        Examples:
            >>> config = AIConfig.from_env()  # Auto-detect
            >>> config = AIConfig.from_env(AIProvider.OPENAI)  # Force OpenAI
        """
        if provider:
            # User specified provider explicitly
            config = cls._create_for_provider(provider)
            if config:
                logger.info(f"Using AI provider: {provider.value}")
                return config
            else:
                logger.warning(f"Provider {provider.value} not available, disabling AI")
                return cls(enabled=False)

        # Auto-detect from environment
        available = detect_available_providers()

        if not available:
            logger.info("No AI providers detected in environment, AI features disabled")
            return cls(enabled=False)

        # Prefer order: OpenAI > Anthropic > Ollama
        for prov in [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA]:
            if prov in available:
                config = cls._create_for_provider(prov)
                if config:
                    logger.info(f"Auto-detected AI provider: {prov.value}")
                    return config

        # Fallback
        logger.info("AI providers found but couldn't configure, disabling AI")
        return cls(enabled=False)

    @classmethod
    def from_function(cls, llm_function: Callable[[str], str], **kwargs) -> "AIConfig":
        """
        Create AIConfig with a custom LLM function.

        This allows using any LLM without configuring a specific provider.

        Args:
            llm_function: Function that takes a prompt (str) and returns response (str)
            **kwargs: Additional config parameters

        Returns:
            AIConfig instance

        Examples:
            >>> def my_llm(prompt: str) -> str:
            ...     return my_model.generate(prompt)
            >>> config = AIConfig.from_function(my_llm)
        """
        config = cls(
            enabled=True,
            provider=AIProvider.CUSTOM,
            **kwargs
        )
        config._llm_function = llm_function
        logger.info("Using custom LLM function")
        return config

    @classmethod
    def disabled(cls) -> "AIConfig":
        """
        Create a disabled AIConfig.

        Returns:
            AIConfig with enabled=False

        Examples:
            >>> config = AIConfig.disabled()
        """
        return cls(enabled=False)

    @classmethod
    def _create_for_provider(cls, provider: AIProvider) -> Optional["AIConfig"]:
        """Create config for a specific provider from environment."""
        if provider == AIProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return cls(
                    enabled=True,
                    provider=provider,
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=api_key,
                    base_url=os.getenv("OPENAI_BASE_URL"),
                )

        elif provider == AIProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return cls(
                    enabled=True,
                    provider=provider,
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    api_key=api_key,
                    base_url=os.getenv("ANTHROPIC_BASE_URL"),
                )

        elif provider == AIProvider.OLLAMA:
            # Ollama doesn't require API key, just check if base URL is set
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
            return cls(
                enabled=True,
                provider=provider,
                model=model,
                base_url=base_url,
            )

        return None

    def get_llm_function(self) -> Callable[[str], str]:
        """
        Get an LLM function based on configuration.

        Returns:
            Callable that takes a prompt and returns a response

        Raises:
            ValueError: If AI is disabled or no provider configured
            ImportError: If required library not installed

        Examples:
            >>> config = AIConfig.from_env()
            >>> llm = config.get_llm_function()
            >>> response = llm("What is 2+2?")
        """
        if not self.enabled:
            raise ValueError("AI features are disabled")

        # Use custom function if provided
        if self._llm_function:
            return self._llm_function

        # Create provider-specific function
        if self.provider == AIProvider.OPENAI:
            return self._create_openai_function()
        elif self.provider == AIProvider.ANTHROPIC:
            return self._create_anthropic_function()
        elif self.provider == AIProvider.OLLAMA:
            return self._create_ollama_function()
        else:
            raise ValueError(f"No LLM function available for provider: {self.provider}")

    def _create_openai_function(self) -> Callable[[str], str]:
        """Create LLM function for OpenAI."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        def llm_function(prompt: str) -> str:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for web content analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.extra_params
            )
            return response.choices[0].message.content

        return llm_function

    def _create_anthropic_function(self) -> Callable[[str], str]:
        """Create LLM function for Anthropic."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        def llm_function(prompt: str) -> str:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **self.extra_params
            )
            return message.content[0].text

        return llm_function

    def _create_ollama_function(self) -> Callable[[str], str]:
        """Create LLM function for Ollama."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Requests library required for Ollama. Install with: pip install requests"
            )

        def llm_function(prompt: str) -> str:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens,
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['response']

        return llm_function

    def __repr__(self) -> str:
        """String representation (hides API key)."""
        if not self.enabled:
            return "AIConfig(enabled=False)"

        api_key_display = "***" if self.api_key else None
        return (
            f"AIConfig(provider={self.provider.value if self.provider else None}, "
            f"model={self.model}, "
            f"api_key={'***' if self.api_key else None})"
        )


def detect_available_providers() -> List[AIProvider]:
    """
    Detect which AI providers are available from environment variables.

    Returns:
        List of available AIProvider enums

    Examples:
        >>> providers = detect_available_providers()
        >>> if AIProvider.OPENAI in providers:
        ...     print("OpenAI is available")
    """
    available = []

    if os.getenv("OPENAI_API_KEY"):
        available.append(AIProvider.OPENAI)

    if os.getenv("ANTHROPIC_API_KEY"):
        available.append(AIProvider.ANTHROPIC)

    # Ollama is "available" if base URL is set or running locally
    if os.getenv("OLLAMA_BASE_URL") or _check_ollama_running():
        available.append(AIProvider.OLLAMA)

    return available


def _check_ollama_running() -> bool:
    """Check if Ollama is running on default port."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_default_llm_function() -> Optional[Callable[[str], str]]:
    """
    Get default LLM function by auto-detecting from environment.

    This is a convenience function for quick usage without explicit config.

    Returns:
        LLM function or None if no provider available

    Examples:
        >>> llm = get_default_llm_function()
        >>> if llm:
        ...     response = llm("Hello!")
    """
    try:
        config = AIConfig.from_env()
        if config.enabled:
            return config.get_llm_function()
    except Exception as e:
        logger.debug(f"Could not create default LLM function: {e}")

    return None
