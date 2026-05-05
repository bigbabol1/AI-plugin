# Adding a language to AI Plugin

## TL;DR
1. Copy `i18n/en.yaml` to `i18n/<code>.yaml` (ISO 639-1 lowercase, e.g. `nl`, `it`, `sv`).
2. Translate every value. Keep `meta.code` equal to the filename stem.
3. Submit a PR.

## Detail
- **Templates** contain `{placeholder}` slots — keep them. Word order can change to fit grammar; just preserve the placeholders.
- **Keyword lists** must be lowercased — the engine matches case-insensitively, but the YAML stores phrases lowercase for consistency.
- Punctuation, accents, and unicode are fine and encouraged where natural.
- **Patterns** (the regex escape hatch) — leave empty unless you NEED a regex the keyword list cannot express. Test it locally first.

## Testing your file
~~~bash
python -c "from custom_components.ai_plugin.i18n import L, SUPPORTED_LANGS; print(SUPPORTED_LANGS)"
~~~
Should list your new code without raising. If it raises, the schema validation message tells you what is wrong.

## Approval bar
PRs welcome. Maintainer reviews for accuracy + grammar + style consistency. Do not bundle other changes.
