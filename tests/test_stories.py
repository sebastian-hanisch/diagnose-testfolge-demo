"""Abnahmekriterien der Presets (dtf_stories): erfüllt von der Messreihe, und jedes Kriterium kippt einzeln an seiner Schwelle (künstliche Werte)."""

import copy

import pytest

import dtf_constants as C
import dtf_results as R
import dtf_stories as ST

RES = R.load()


def test_every_preset_has_a_story_and_it_holds_on_the_measurement():
    assert set(ST.STORIES) == set(C.PRESETS)
    assert all(ST.check_all(RES).values()), ST.check_all(RES)


def _fake(cid, **changes):
    res = copy.deepcopy(RES)
    cell = res["cells"][cid]
    for path, value in changes.items():
        target = cell
        *head, last = path.split(".")
        for h in head:
            target = target[h]
        target[last] = value
    return res


def test_standard_story_flips_when_the_order_of_the_rules_changes():
    assert ST.story_standard(RES)
    assert not ST.story_standard(_fake("q2", **{"M.mean": 30.0}))
    assert not ST.story_standard(_fake("q2", **{"G.mean": 40.0}))                         # G über B


def test_without_noise_story_flips_at_the_threshold_of_three_percent():
    assert ST.story_ohne_rauschen(_fake("q0", **{"M.mean": 2.9, "G.mean": 2.9, "B.mean": 2.9}))
    assert not ST.story_ohne_rauschen(_fake("q0", **{"M.mean": 3.0, "G.mean": 3.0, "B.mean": 3.0}))


def test_good_tests_story_flips_at_ten_and_twentyfive_percent():
    assert ST.story_gute_tests(_fake("q1", **{"G.mean": 10.0, "B.mean": 25.0}))
    assert not ST.story_gute_tests(_fake("q1", **{"G.mean": 9.9, "B.mean": 25.0}))
    assert not ST.story_gute_tests(_fake("q1", **{"G.mean": 10.0, "B.mean": 24.9}))


@pytest.mark.parametrize("cid,fn", [("q5", ST.story_teurer_irrtum), ("g1", ST.story_alle_gleich)])
def test_sign_stories_need_more_than_two_standard_errors(cid, fn):
    assert fn(RES)
    assert fn(_fake(cid, **{"diff_MG.mean": 2.01, "diff_MG.se": 1.0}))
    assert not fn(_fake(cid, **{"diff_MG.mean": 2.0, "diff_MG.se": 1.0}))                 # genau zwei Standardfehler genügen nicht
    assert not fn(_fake(cid, **{"diff_MG.mean": -5.0, "diff_MG.se": 0.1}))                # Vorzeichen falsch


def test_genome_story_flips_at_twenty_and_sixty_percent():
    assert ST.story_genom(_fake("y50c2", share_first=0.2)) and ST.story_genom(_fake("y50c2", share_first=0.6))
    assert not ST.story_genom(_fake("y50c2", share_first=0.19)) and not ST.story_genom(_fake("y50c2", share_first=0.61))
