from unittest import mock

import pytest
from IPython.core import display_functions

from genai.display import GenaiMarkdown, Stage


def test_genai_markdown_init_default(ip):
    markdown = GenaiMarkdown()

    assert markdown.message == " "
    assert markdown.stage is None


def test_genai_markdown_init_with_args(ip):
    markdown = GenaiMarkdown(message="Hello world!", stage=Stage.GENERATING)

    assert markdown.message == "Hello world!"
    assert markdown.stage == Stage.GENERATING


def test_genai_markdown_append(ip):
    markdown = GenaiMarkdown(message="Hello")
    markdown.append(" world!")

    assert markdown.message == "Hello world!"


def test_genai_markdown_consume(ip):
    markdown = GenaiMarkdown(message="Hello")

    def text_generator():
        yield " world"
        yield "!"

    markdown.consume(text_generator())

    assert markdown.message == "Hello world!"


def test_genai_markdown_display(ip):
    markdown = GenaiMarkdown(message="Hello world!")

    with mock.patch.object(display_functions, 'display') as mock_display:
        markdown.display()
        mock_display.assert_called_once_with(markdown, display_id=markdown._display_id)


def test_genai_markdown_update_displays(ip):
    markdown = GenaiMarkdown(message="Hello world!")

    with mock.patch.object(display_functions, 'display') as mock_display:
        markdown.update_displays()
        mock_display.assert_called_once_with(markdown, display_id=markdown._display_id, update=True)


def test_genai_markdown_repr(ip):
    markdown = GenaiMarkdown(message="Hello world!")

    assert repr(markdown) == "Hello world!"


def test_genai_markdown__repr_markdown_without_stage(ip):
    markdown = GenaiMarkdown(message="Hello world!")

    result = markdown._repr_markdown_()
    assert result == "Hello world!"


def test_genai_markdown__repr_markdown_with_stage(ip):
    markdown = GenaiMarkdown(message="Hello world!", stage=Stage.GENERATING)

    result = markdown._repr_markdown_()
    assert result == ("Hello world!", {"genai": {"stage": Stage.GENERATING}})


def test_genai_markdown_message_setter(ip):
    markdown = GenaiMarkdown(message="Hello")

    with mock.patch.object(markdown, 'update_displays') as mock_update_displays:
        markdown.message = "Hello world!"
        mock_update_displays.assert_called_once()
        assert markdown.message == "Hello world!"


def test_genai_markdown_stage_setter(ip):
    markdown = GenaiMarkdown(stage=None)
    markdown.stage = Stage.GENERATING

    assert markdown.stage == Stage.GENERATING
