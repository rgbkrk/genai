# 👨‍🔬👨🏿‍🔬 GenAI Changelog 👩🏾‍🔬👩‍🔬

---

## Unreleased

#### Enhancements:

- **Code refactoring**: Improved code organization and readability by extracting functions for crafting messages and generating smaller reprs with `craft_message`, `repr_genai_pandas`, and `repr_genai`. 🧰
- **Enhanced pandas support**: Optimized DataFrame and Series representation for GPT-3 and GPT-4 by using Markdown format. 📊
- **Token management**: Introduced a new module `tokens.py` with utility functions `num_tokens_from_messages` and `trim_messages_to_fit_token_limit` to handle token count and message trimming based on model limitations and your wallet. 💸

#### Changes:

- `craft_user_message` now uses the new `craft_message` function.
- `craft_output_message` now uses the new `repr_genai` function.
- The `get_historical_context` function now accepts an additional `model` parameter to support different GPT models and has been updated to use `tokens.trim_messages_to_fit_token_limit`.
- The `ignore_tokens` list now uses the term "first line" instead of "start" for clarity.
- Introduced support for GPT-4 token counting and message trimming in `tokens.py`.

#### Bug Fixes:

- N/A

## `1.0.3`

Fixed

- Sample Pandas Series properly
- Include `tabulate` dependency for markdown conversion

## `1.0.2`

### Fixed

- Correctly lowered text formatted context size on `DataFrame`s and `Series` for `%%assist` magic command

## `1.0.1`

### Changed

- Updated README.md

## `1.0.0`

_2023-03-14_

🎉 Initial release 🎉

### Added

- Custom exception suggestions
- `%%assist` magic command to generate code from natural language
