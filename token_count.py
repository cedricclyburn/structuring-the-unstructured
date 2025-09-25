"""Token counting utility for OpenAI models."""

import tiktoken


class TokenCount:
    """A class to count tokens for OpenAI models using tiktoken."""
    
    def __init__(self, model_name="gpt-4o-turbo"):
        """Initialize the TokenCount with a specific model.
        
        Args:
            model_name: The name of the OpenAI model for token counting
        """
        self.model_name = model_name
        # Map model names to their encoding names
        encoding_name = self._get_encoding_for_model(model_name)
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _get_encoding_for_model(self, model_name):
        """Get the appropriate encoding for the model."""
        # Handle different model name variations
        if "gpt-4" in model_name:
            return "cl100k_base"  # GPT-4 uses this encoding
        elif "gpt-3.5" in model_name:
            return "cl100k_base"  # GPT-3.5-turbo also uses this
        else:
            # Default to cl100k_base for most modern models
            return "cl100k_base"
    
    def num_tokens_from_string(self, text):
        """Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
        """
        if not text:
            return 0
        tokens = self.encoding.encode(text)
        return len(tokens)
