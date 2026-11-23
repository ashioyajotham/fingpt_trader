import io
import json
import os
import socket
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from fingpt_research.cli import main


class OfflineReplay(unittest.TestCase):
    def replay(self, *args):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            socket, "socket", side_effect=AssertionError("network access")
        ), redirect_stdout(output):
            main(list(args))
        return output.getvalue()

    def test_default_replay(self):
        first = self.replay()
        self.assertEqual(first, self.replay())
        bars = json.loads(first)["observations"]
        self.assertEqual(len(bars), 12)
        self.assertEqual(bars[0]["close"], "94591.79000000")
        early = self.replay("--as-of", "2025-01-01T12:00:00Z")
        self.assertEqual(json.loads(early)["observations"], [])
