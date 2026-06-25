#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for message parsing."""

import pytest
from src.parser import extract_messages_from_text, validate_whatsapp_file


class TestParserAndroid:
    """Android format tests."""

    def test_should_extract_messages_when_android_format(self):
        """Extract messages from Android format export."""
        content = """12/06/24, 10:30 a. m. - Alice: Hello
12/06/24, 10:31 a. m. - Bob: Hi there"""

        messages = extract_messages_from_text(content)
        assert len(messages) == 2
        assert messages[0][1] == "Alice"
        assert messages[0][2] == "Hello"
        assert messages[1][1] == "Bob"
        assert messages[1][2] == "Hi there"

    def test_should_skip_system_messages(self):
        """Filter out system messages from exports."""
        content = """12/06/24, 10:30 a. m. - Alice: Hello
12/06/24, 10:31 a. m. - <multimedia omitido>
12/06/24, 10:32 a. m. - Bob: Hi there"""

        messages = extract_messages_from_text(content)
        assert len(messages) == 2
        assert messages[0][1] == "Alice"
        assert messages[1][1] == "Bob"

    def test_should_handle_multiline_messages(self):
        """Support multiline messages in conversation."""
        content = """12/06/24, 10:30 a. m. - Alice: First line
second line
12/06/24, 10:31 a. m. - Bob: Single line"""

        messages = extract_messages_from_text(content)
        assert len(messages) == 2
        assert "second line" in messages[0][2]


class TestValidation:
    """File validation tests."""

    def test_should_validate_whatsapp_format(self):
        """Recognize valid WhatsApp export format."""
        content = """12/06/24, 10:30 a. m. - Alice: Hello
12/06/24, 10:31 a. m. - Bob: Hi"""

        is_valid, _ = validate_whatsapp_file(content)
        assert is_valid

    def test_should_reject_invalid_format(self):
        """Reject files that don't match WhatsApp pattern."""
        content = """This is not a valid WhatsApp export
Just random text here"""

        is_valid, _ = validate_whatsapp_file(content)
        assert not is_valid
