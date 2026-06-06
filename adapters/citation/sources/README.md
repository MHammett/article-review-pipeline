# Citation Source Adapter Interface

Each source adapter must expose a `resolve(claim, api_key=None)` function and return a dict:

- `found` (bool): whether a relevant primary source was located
- `url` (str): URL to the primary source (present when found=True)
- `summary` (str): human-readable summary of what was found
- `content` (str): raw content used for SHA-256 checksum generation

To add a source adapter:
1. Create a new file in adapters/citation/sources/
2. Implement `resolve(claim, api_key=None)`
3. Register the adapter name in adapters/citation/resolver.py ADAPTER_MAP
4. Add it to your publication config under citation_sources with the adapter name
