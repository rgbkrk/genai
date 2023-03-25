# 👨‍🔬👨🏿‍🔬 GenAI Changelog 👩🏾‍🔬👩‍🔬

---

## Unreleased

## `2.0.0`

### Enhancements:

#### Added

- 🔄 Keep conversations flowing with `%%assist` (#66)
- 🖼️ Emit suggestions as `Markdown` instead of creating new cells (#66)
- 🚀 Model selection made easy with the `--model` flag for `%%assist` (#65)
- 💡 Introducing `GenaiMarkdown` – a dynamic Markdown display (#61)
- 📝 Create a `%%prompt` magic for setting the default prompts for assistance and exceptions (#71, #69)

#### Changed

- 🧪 Craft a more ipythonic context manager (#62, #66)
  - Meet the new `Context` class: capture IPython history and make it ChatCompletion-friendly
  - Farewell `get_historical_context`, hello `build_context`: context construction using the new Context class
  - Reduce messages sent to GPT models by trimming based on estimated number of tokens (#57)

- 🎯 Type annotations step in! (#59)


#### Improved

- 📏 Token length checks now available in %%assist (#57)
- 🧹 Code refactoring: introducing `craft_message`, `repr_genai_pandas`, and `repr_genai` for more organized and readable code
- 📈 Enhanced pandas support: optimized DataFrame and Series representation for Large Language Model consumption using Markdown format
- 💰 Token management: a new module `tokens.py` featuring `num_tokens_from_messages` and `trim_messages_to_fit_token_limit` to help you stay within model limitations and budget
- 📚 Update assist magic documentation (#70)

#### Removed

- 🚫 `%%assist` no longer generates new code cells. It now creates Markdown output instead (#66)
  - Relatedly, `in-place` is no longer an option since we do not change the cells


### Changes:

- `craft_user_message` now relies on the new `craft_message` function
- `craft_output_message` has been upgraded to use the new `repr_genai` function
- `get_historical_context` now sports an additional `model` parameter and utilizes `tokens.trim_messages_to_fit_token_limit`
- For clarity, the `ignore_tokens` list now uses the term "first line" instead of "start"
- GPT-4 token counting and message trimming now supported in `tokens.py`

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
