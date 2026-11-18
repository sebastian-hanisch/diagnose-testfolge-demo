"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dtf_model as M  # noqa: E402


def make_problem(prior, tests, costs, loss):
    """Problem aus Vorab, Tests als Liste von Ausgangstabellen (Zeilen = Ausgänge, Spalten = Diagnosen), Kosten und Irrtumspreis."""
    return M.Problem(prior, [np.array(t, float) for t in tests], costs, loss)


@pytest.fixture
def one_test():
    """Zwei Diagnosen je 1/2, ein Test mit Se = Sp = 0,9 (positiv: Diagnose 1 mit 0,9, Diagnose 2 mit 0,1), Kosten 1, Irrtumspreis 10.
    Handrechnung: ohne Test 10 · 1/2 = 5; mit Test 1 + 10 · 0,1 = 2; Optimum 2."""
    return make_problem([0.5, 0.5], [[[0.1, 0.9], [0.9, 0.1]]], [1.0], 10.0)


@pytest.fixture
def two_equal_tests():
    """Wie one_test, aber zwei gleiche Tests (Kosten je 1) und Irrtumspreis 100.
    Handrechnung: sofort 50; ein Test 1 + 100 · 0,1 = 11; zwei Tests 2 + (0,82 · 100/82 + 0,18 · 50) = 12; Optimum 11 (nach einem Test stoppen)."""
    t = [[0.1, 0.9], [0.9, 0.1]]
    return make_problem([0.5, 0.5], [t, t], [1.0, 1.0], 100.0)
