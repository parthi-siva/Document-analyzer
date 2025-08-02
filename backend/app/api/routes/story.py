from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from openai import OpenAI
import os
import re
import logging

from app.services.prompt_manager import (
    get_story_generation_prompt,
    get_story_validation_prompt,
    get_model_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)

router = APIRouter()

@router.get("/generate-story")
async def generate_story(
    genre: str = Query(
        ..., description="Genre of the story (e.g., fantasy, sci-fi, romance)"
    ),
    characters: Union[int, str] = Query(
        0,
        description="Number of characters in the story (e.g., 3 for three characters) or a comma-separated list of names",
    ),
    paragraph_length: int = Query(
        5, description="Length of each paragraph in the story (number of sentences)"
    ),
) -> Any:
    logger.info(
        f"generate_story called with genre={genre}, characters={characters}, paragraph_length={paragraph_length}"
    )

    # Handle characters as int or comma-separated string
    if isinstance(characters, int):
        char_desc = f"{characters} characters"
    else:
        # If string, count names separated by commas
        char_list = [c.strip() for c in str(characters).split(",") if c.strip()]
        char_desc = f"characters: {', '.join(char_list)}"

    # Get prompts using the prompt manager
    system_prompt, user_prompt = get_story_generation_prompt(genre, char_desc, paragraph_length)
    logger.info(f"Prompt for story generation: {user_prompt}")

    # Get model configuration
    model_config = get_model_config()

    logger.info("Sending request to OpenAI for story generation...")
    response = openai.chat.completions.create(
        model=model_config.get('model_name', 'Qwen/Qwen3-32B'),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=model_config.get('temperature', 0.7),
        top_p=model_config.get('top_p', 0.8),
        max_tokens=model_config.get('max_tokens', 1024),
    )
    logger.info("Received response from OpenAI for story generation.")

    content = response.choices[0].message.content
    logger.info(f"Raw story content: {content}")

    if content is not None:
        story = content.strip()
    else:
        story = ""
    # Remove everything between <think> and </think> (including the tags)
    story = re.sub(r"<think>.*?</think>\s*", "", story, flags=re.DOTALL | re.IGNORECASE)
    logger.info(f"Story after removing <think> tags: {story}")

    # Extract title (expects format: **Title: ...**)
    title_match = re.match(r"\*\*\s*\s*(.*?)\*\*", story)
    if title_match:
        title = title_match.group(1).strip()
        # Remove the title line from the story
        story_body = story[title_match.end() :].lstrip()
        logger.info(f"Extracted title: {title}")
    else:
        title = ""
        story_body = story
        logger.warning("Title not found in story.")

    logger.info("Preparing to validate the generated story...")
    validation_system_prompt = get_story_validation_prompt(genre, str(characters), paragraph_length)
    logger.info(f"Validation system prompt: {validation_system_prompt}")
    logger.info(f"Story body sent for validation: {story_body}")

    response = openai.chat.completions.create(
        model=model_config.get('model_name', 'Qwen/Qwen3-32B'),
        messages=[
            {"role": "system", "content": validation_system_prompt},
            {"role": "user", "content": story_body},
        ],
        temperature=model_config.get('temperature', 0.7),
        top_p=model_config.get('top_p', 0.8),
        max_tokens=model_config.get('max_tokens', 1024),
    )
    logger.info("Received response from OpenAI for validation.")

    validation_response = response.choices[0].message.content
    logger.info(f"Raw validation response: {validation_response}")

    validation_response = validation_response.strip() if validation_response else ""
    response_clean = re.sub(
        r"<think>.*?</think>\s*",
        "",
        validation_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    logger.info(f"Validation response after removing <think> tags: {response_clean}")

    if response_clean.lower() != "y":
        logger.error("The generated story does not meet the specified criteria.")
        raise HTTPException(
            status_code=400,
            detail="The generated story does not meet the specified criteria.",
        )

    logger.info(f"Returning story with title: {title}")
    return {"title": title, "story": story_body}
