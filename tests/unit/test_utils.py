"""Tests for mira.core.utils module."""

from mira.core import (
    get_mira_path, extract_text_content, parse_timestamp, get_custodian
)


class TestUtils:
    """Test utility functions."""

    def test_extract_text_content_string(self):
        msg = {'content': 'Hello world'}
        assert extract_text_content(msg) == 'Hello world'

    def test_extract_text_content_list(self):
        msg = {'content': [
            {'type': 'text', 'text': 'Part 1'},
            {'type': 'text', 'text': 'Part 2'}
        ]}
        assert extract_text_content(msg) == 'Part 1\nPart 2'

    def test_extract_text_content_empty(self):
        assert extract_text_content({}) == ''
        assert extract_text_content(None) == ''

    def test_parse_timestamp_valid(self):
        ts = parse_timestamp('2025-12-07T14:30:00.123Z')
        assert ts is not None
        assert ts.year == 2025
        assert ts.month == 12
        assert ts.day == 7

    def test_parse_timestamp_invalid(self):
        assert parse_timestamp('') is None
        assert parse_timestamp('invalid') is None
        assert parse_timestamp(None) is None

    def test_get_mira_path(self):
        path = get_mira_path()
        assert path is not None
        assert str(path).endswith('.mira')

    def test_get_custodian(self):
        custodian = get_custodian()
        assert custodian is not None
        assert isinstance(custodian, str)
        assert len(custodian) > 0
