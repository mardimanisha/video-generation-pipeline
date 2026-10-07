from services.script_analyzer import analyze, split_long_sentence, split_paragraphs, split_sentences


def test_sentences_respect_abbreviations_decimals_and_initials():
    text = "Dr. Smith measured 3.5 litres, e.g. in the lab. J. K. Rowling agreed! Did it work? Yes."
    assert split_sentences(text) == [
        "Dr. Smith measured 3.5 litres, e.g. in the lab.",
        "J. K. Rowling agreed!",
        "Did it work?",
        "Yes.",
    ]


def test_quotes_stay_with_their_sentence():
    assert split_sentences('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]


def test_paragraphs_and_headings():
    script = "Introduction\nThis is line one\nand its continuation.\n\nSecond paragraph here."
    assert split_paragraphs(script) == [
        "Introduction",
        "This is line one and its continuation.",
        "Second paragraph here.",
    ]


def test_windows_newlines_and_bom():
    assert split_paragraphs("﻿A.\r\n\r\nB.") == ["A.", "B."]


def test_long_sentence_splits_only_at_clause_boundaries():
    s = ("When you start investing early in your career, even small monthly contributions, "
         "which may seem insignificant at first, can grow into a remarkably large sum over several decades.")
    parts = split_long_sentence(s, max_duration=5.0, wps=2.5)
    assert len(parts) > 1
    assert " ".join(parts) == s  # nothing lost, no word broken
    assert all(len(p.split()) >= 3 for p in parts)


def test_unsplittable_sentence_is_kept_whole():
    s = "Supercalifragilistic " * 30
    assert split_long_sentence(s.strip(), max_duration=3.0, wps=2.5) == [s.strip()]


def test_analyze_preserves_all_text():
    script = "One two three. Four five six?\n\nSeven eight, nine ten."
    units = analyze(script, 2.5, 8.0)
    assert [u.text for u in units] == ["One two three.", "Four five six?", "Seven eight, nine ten."]
    assert [u.paragraph for u in units] == [0, 0, 1]
    assert all(u.estimate > 0 for u in units)
