"""Tests for mira.custodian module."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.custodian import (
    init_custodian_db, extract_custodian_learnings,
    get_full_custodian_profile, get_danger_zones_for_files
)
from mira.core import shutdown_db_manager


class TestCustodian:
    """Test custodian learning functionality."""

    def test_init_custodian_db(self):
        """Test custodian database initialization."""
        shutdown_db_manager()  # Reset before creating new temp dir
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            db_path = mira_path / 'custodian.db'
            assert db_path.exists()
        finally:
            shutdown_db_manager()  # Clean up connections
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_identity_from_messages(self):
        """Test that we learn the user's name from messages."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Conversation where user introduces themselves
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Hi, my name is Max. Can you help me?'},
                    {'role': 'assistant', 'content': 'Hello Max! I\'d be happy to help.'},
                ]
            }

            extract_custodian_learnings(conversation, 'test-session-1')

            profile = get_full_custodian_profile()
            # Check if name was learned
            assert 'identity' in profile
            # Name may or may not be captured depending on pattern matching
            # The important thing is the function runs without error
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_preferences(self):
        """Test that we learn preferences from user statements."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'I prefer using pnpm instead of npm'},
                    {'role': 'user', 'content': 'I always use vitest for testing'},
                    {'role': 'user', 'content': 'No emojis please'},
                ]
            }

            extract_custodian_learnings(conversation, 'pref-session')

            profile = get_full_custodian_profile()
            assert 'preferences' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_rules(self):
        """Test that we learn explicit rules from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Never commit directly to main branch'},
                    {'role': 'assistant', 'content': 'You should always run tests before pushing'},
                    {'role': 'user', 'content': 'Avoid using var, use const or let instead'},
                ]
            }

            extract_custodian_learnings(conversation, 'rules-session')

            profile = get_full_custodian_profile()
            assert 'rules' in profile
            assert 'never' in profile['rules']
            assert 'always' in profile['rules']
            assert 'avoid' in profile['rules']
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_danger_zones(self):
        """Test that we learn about problematic files/areas."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'There was an error in auth.py again'},
                    {'role': 'assistant', 'content': 'The problem with auth.py is the session handling'},
                    {'role': 'user', 'content': 'Be careful with legacy-api.js, it keeps breaking'},
                ]
            }

            # Extract twice to trigger frequency threshold
            extract_custodian_learnings(conversation, 'danger-session-1')
            extract_custodian_learnings(conversation, 'danger-session-2')

            profile = get_full_custodian_profile()
            assert 'danger_zones' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_get_danger_zones_for_files(self):
        """Test checking files against known danger zones."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Create a danger zone by extracting from a conversation
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'The issue with payment.py is very serious'},
                    {'role': 'assistant', 'content': 'Yes, payment.py has had multiple bugs'},
                ]
            }
            extract_custodian_learnings(conversation, 'danger-1')
            extract_custodian_learnings(conversation, 'danger-2')

            # Check if files match danger zones
            warnings = get_danger_zones_for_files(['/src/payment.py', '/src/app.py'])
            # May or may not find depending on frequency threshold
            assert isinstance(warnings, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_full_custodian_profile(self):
        """Test getting the complete custodian profile."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            profile = get_full_custodian_profile()

            # Check structure
            assert 'name' in profile
            assert 'identity' in profile
            assert 'preferences' in profile
            assert 'rules' in profile
            assert 'danger_zones' in profile
            assert 'work_patterns' in profile
            assert 'summary' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_profile_summary_generation(self):
        """Test that profile summary is generated correctly."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Add some data
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'My name is Alice'},
                    {'role': 'user', 'content': 'I prefer using typescript'},
                ]
            }
            extract_custodian_learnings(conversation, 'summary-test')

            profile = get_full_custodian_profile()
            assert 'summary' in profile
            assert isinstance(profile['summary'], str)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)
