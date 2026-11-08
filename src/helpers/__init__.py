"""Shared helpers for the lesson scripts. Import them by module, e.g. `from helpers.models import CHEAPEST`.

    models.py            model aliases (CHEAPEST, BALANCED, MOST_CAPABLE, ALL); loads .env
    log_helper.py        prints what is sent to and received from Claude
    chat_helper.py       add_user_message, add_assistant_message, send, chat
    prompt_evaluator.py  the eval pipeline used by the prompt engineering lessons (15-19)
"""
