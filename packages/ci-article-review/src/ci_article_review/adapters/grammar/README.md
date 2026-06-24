# Grammar Adapter Interface

Each grammar adapter must expose a `run(text, lt_config, username, api_key, retry, retry_delay)` function
and return a dict with these keys:

- `corrected_text` (str): draft after auto-applied corrections
- `change_log` (list): each entry has `rule_id`, `category`, `original`, `replacement`, `offset`, `message`
- `flagged_matches` (list): matches in `flag_for_review` categories, not auto-applied
- `failed` (bool): True if the API call failed entirely
- `error` (str, optional): error message if failed
