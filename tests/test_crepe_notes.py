#!/usr/bin/env python

"""Tests for `crepe_notes` package."""


import unittest
from click.testing import CliRunner

from crepe_notes import crepe_notes
from crepe_notes import cli


class TestCrepe_notes(unittest.TestCase):
    """Tests for `crepe_notes` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'crepe_notes.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
