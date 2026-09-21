# Design archive

Historical planning and specification documents, written before the features
below were built. They are kept for the reasoning they record, not as
documentation: **they were accurate when written and are not maintained.**

Where an archived document and the shipped code disagree, the code wins, and
`CHANGELOG.md` records what actually shipped.

| Document | Shipped in |
|---|---|
| `plans/2026-04-22-set-area-state.md`, `specs/2026-04-22-set-area-state-design.md` | `set_area_state` tool |
| `plans/2026-05-03-multilang-prompts.md`, `specs/2026-05-03-multilang-prompts-design.md` | per-language prompt hints (later replaced by the YAML i18n layer) |
| `plans/2026-05-05-multilang-data-driven.md`, `specs/2026-05-05-multilang-data-driven-design.md` | `custom_components/ai_plugin/i18n/` |
| `plans/2026-05-05-sat1-eval-harness.md`, `specs/2026-05-05-sat1-eval-harness-design.md` | `tests/eval/` |

Known drift: the multilang documents still show the `fallback_no_sensor`
template and the `area_prefixes` keyword list. Both were specified, shipped,
never read by any code path, and removed in v0.9.53.
