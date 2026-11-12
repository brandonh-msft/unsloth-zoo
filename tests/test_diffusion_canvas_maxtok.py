# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import io
import sys

import pytest

from unsloth_zoo.diffusion_studio import visual_engine as V


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),          # auto-size sentinel stays auto
        (1, 1),          # small explicit honored
        (4096, 4096),
        (8192, 8192),
        (8193, 8193),
        (24576, 24576),
        (65535, 65535),
        (262144, 262144),
    ],
)
def test_canvas_maxtok_preserves_explicit_value(value, expected):
    assert V._canvas_maxtok(value) == expected


def test_canvas_maxtok_coerces_str():
    assert V._canvas_maxtok("4096") == 4096
    assert V._canvas_maxtok("99999") == 99999


@pytest.mark.parametrize("ready", ["READY 262144 2048\n", "READY 262144\n"])
def test_unverified_or_mismatched_explicit_context_reaps_child(monkeypatch, tmp_path, ready):
    class Child:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.TextIOWrapper(io.BytesIO(ready.encode("utf-8")))
            self.killed = False
            self.waited = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            self.waited = True
            return 0

    child = Child()
    monkeypatch.setattr(V.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match = "context"):
        V.VisualServer("model.gguf", server_bin = sys.executable, maxtok = 4096)
    assert child.killed and child.waited


def test_old_runner_is_rejected_before_a_large_context_is_spawned(monkeypatch):
    result = type("Result", (), {"returncode": 0, "stdout": "multi_gpu_split\n"})()
    monkeypatch.setattr(V.subprocess, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        V.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("an unsupported runner was started"),
    )
    with pytest.raises(RuntimeError, match = "8192"):
        V.VisualServer("model.gguf", server_bin = sys.executable, maxtok = 24576)


def test_new_runner_receives_and_reports_the_exact_large_context(monkeypatch, tmp_path):
    result = type("Result", (), {"returncode": 0, "stdout": "multi_gpu_split exact_context_v1\n"})()
    monkeypatch.setattr(V.subprocess, "run", lambda *_args, **_kwargs: result)
    seen = {}

    class Child:
        def __init__(self, *_args, **kwargs):
            seen["maxtok"] = kwargs["env"]["MAXTOK"]
            self.stdout = io.TextIOWrapper(io.BytesIO(b"READY 262144 65535\n"))

    monkeypatch.setattr(V.subprocess, "Popen", Child)
    server = V.VisualServer(
        "model.gguf", server_bin = sys.executable, maxtok = 65535,
        req_path = str(tmp_path / "request.json"),
    )
    assert seen["maxtok"] == "65535"
    assert server.maxtok == 65535
