# Tests

Run all tests:
```
pytest tests/
```

`test_adapters.py` — unit tests for LanguageTool, OpenAI, Mistral, and Gemini adapters.
All external HTTP calls are mocked; no API keys required.

`test_consolidation.py` — unit tests for consensus detection, delta computation, and report building.
