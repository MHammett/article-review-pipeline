# Citation Source Adapter Interface

Each source adapter must expose a `resolve(claim, api_key=None)` function and return a dict:

- `found` (bool): whether a relevant primary source was located
- `url` (str): URL to the primary source (present when found=True)
- `summary` (str): human-readable summary of what was found
- `content` (str): raw content used for SHA-256 checksum generation
- `pointer_only` (bool, optional): set True when the adapter is only naming a place a
  human could look this up, rather than retrieving and checking the data itself

`pointer_only` decides the confidence tier the resolver reports. Omit it (or set it
False) only if the adapter genuinely fetched the underlying data — that puts the claim
in the "verified" tier and tells a reader the checksummed content *is* the evidence.
Adapters that match a keyword and hand back a portal URL must set it True; see
[docs/CITATIONS.md](../../../../../../../docs/CITATIONS.md).

Currently: `census`, `crossref`, `eia`, `fred` are data-fetching; `epa`, `ferc`, `fhwa`,
`icc`, `ilga`, `pjm` are pointer-only.

Pointer-only adapters that gate on keyword lists should match through
`topic_match.topic_match()` rather than raw substring containment, so a keyword
appearing in a credential phrase ("credentials in air quality analysis") doesn't
resolve the claim to an unrelated regulatory portal.

To add a source adapter:
1. Create a new file in adapters/citation/sources/
2. Implement `resolve(claim, api_key=None)`
3. Register the adapter name in adapters/citation/resolver.py ADAPTER_MAP
4. Add it to your publication config under citation_sources with the adapter name
5. Document it in the adapter table in docs/CITATIONS.md
