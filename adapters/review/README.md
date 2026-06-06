# Review Adapter Interface

Each review adapter must expose a `call(system_prompt, user_prompt, api_key, retry, retry_delay)` function
and return a dict with these keys:

- `failed` (bool)
- `data` (dict): parsed JSON response from the model (present when failed=False)
- `model` (str): model identifier used
- `tokens` (dict): `{"prompt": int, "completion": int}`
- `error` (str, optional): error message if failed
- `raw` (str, optional): raw response text if JSON parsing failed

To add a new review adapter:
1. Create a new file in adapters/review/
2. Implement the `call()` function with the signature above
3. Register it in the pipeline's review pass configuration
