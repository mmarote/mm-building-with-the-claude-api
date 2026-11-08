"""
The only place model IDs appear in this project.
When a new model version ships, update it here and every script follows.

Importing this module also loads the project's .env file (ANTHROPIC_API_KEY),
so the lessons work from any terminal without exporting the key first.
"""

from dotenv import load_dotenv

load_dotenv()  # looks for .env in this folder, then in parent folders

CHEAPEST = "claude-haiku-5-5"
BALANCED = "claude-sonnet-5-5"
MOST_CAPABLE = "claude-opus-5-5"

ALL = [CHEAPEST, BALANCED, MOST_CAPABLE]
