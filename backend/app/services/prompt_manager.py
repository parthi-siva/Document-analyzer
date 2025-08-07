"""
Prompt Management Service

"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts and templates"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.prompts = {}
        self.load_prompts()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "prompts" / "question_answer_promt.yaml")

    def load_prompts(self) -> None:
        """Load prompts from the configuration file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(
                    ".yml"
                ):
                    self.prompts = yaml.safe_load(file)
                elif self.config_path.endswith(".json"):
                    self.prompts = json.load(file)
                else:
                    raise ValueError(f"Unsupported file format: {self.config_path}")

            logger.info(f"Successfully loaded prompts from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Prompt configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

    def get_prompt(self, category: str, prompt_type: str, **kwargs) -> str:
        """
        Get a formatted prompt.

        Args:
            category: The category of the prompt
            prompt_type: The type of prompt
            **kwargs: Variables to format into the prompt template

        Returns:
            The formatted prompt string

        Raises:
            KeyError: If the prompt category or type is not found
            ValueError: If required template variables are missing
        """
        try:
            prompt_template = self.prompts[category][prompt_type]

            # If it's a template, format it with the provided kwargs
            if "{" in prompt_template and "}" in prompt_template:
                return prompt_template.format(**kwargs)
            else:
                return prompt_template

        except KeyError as e:
            logger.error(
                f"Prompt not found: category='{category}', type='{prompt_type}'"
            )
            raise KeyError(f"Prompt not found: {e}")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration settings.

        Returns:
            Dictionary containing model configuration
        """
        return self.prompts.get("model_config", {})

    def reload_prompts(self) -> None:
        """Reload prompts from the configuration file."""
        self.load_prompts()
        logger.info("Prompts reloaded successfully")

    def list_available_prompts(self) -> Dict[str, list]:
        """
        List all available prompt categories and types.

        Returns:
            Dictionary with categories as keys and lists of prompt types as values
        """
        available = {}
        for category, prompts in self.prompts.items():
            if isinstance(prompts, dict) and category != "model_config":
                available[category] = list(prompts.keys())
        return available


# Singleton instance for easy access
_prompt_manager_instance = None


def get_prompt_manager() -> PromptManager:
    """
    Get the singleton instance of the PromptManager.

    Returns:
        PromptManager instance
    """
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance


# Convenience functions for common operations
def get_question_answer_prompt(context: str, query: str) -> tuple[str, str]:
    """
    Get formatted prompts.

    Args:
        context: Context information for the prompt
        query: User query

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    pm = get_prompt_manager()
    system_prompt = pm.get_prompt("question_answer", "system_prompt")
    user_prompt = pm.get_prompt(
        "question_answer", "user_prompt_template", context_text=context, query=query
    )
    return system_prompt, user_prompt


def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration.

    Returns:
        Model configuration dictionary
    """
    pm = get_prompt_manager()
    return pm.get_model_config()
