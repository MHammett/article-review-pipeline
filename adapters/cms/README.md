# CMS Adapter Interface

Each CMS adapter must expose:

- `push(content, pub_params, cms_config, seo_config, publish_live=False)` → dict with `success`, `post_id`, `post_url`, `error`
- `print_checklist_and_confirm()` → bool (True if user confirmed, False if aborted)

To add a new CMS adapter:
1. Create a new file in adapters/cms/
2. Implement the interface above
3. In your publication config, change the `wordpress` block to `cms` with an `adapter` field
