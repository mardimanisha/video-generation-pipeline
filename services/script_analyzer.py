"""Split a lecture script into paragraphs, sentences and (when needed) clauses.

Splits only at natural speech boundaries: paragraph breaks, sentence ends, and, for a
sentence too long for one clip, clause punctuation or a conjunction. Never inside a word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from utils.duration import count_words, estimate_speech_duration

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig",
    "no", "vol", "approx", "inc", "ltd", "co", "dept", "est", "u.s", "u.k", "a.m", "p.m",
}
# Candidate sentence end: terminal punctuation, optional closing quote/bracket, whitespace, next token.
_BOUNDARY = re.compile(r"([.!?…]+[\"'”’)\]]*)\s+(?=\S)")
_CLAUSE = re.compile(r"(?<=[,;:—])\s+")
_CONJUNCTION = re.compile(r"\s+(?=(?:and|but|so|because|which|while|although|however|whereas)\b)", re.I)


@dataclass
class Sentence:
    text: str
    paragraph: int
    estimate: float


def normalize(script: str) -> str:
    script = script.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in script.split("\n")]
    return "\n".join(lines).strip()


def split_paragraphs(script: str) -> list[str]:
    """Blank lines separate paragraphs (conceptual boundaries). Single newlines are joined,
    except that a short line without terminal punctuation (a heading) stands alone."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", normalize(script)):
        lines = [ln for ln in block.split("\n") if ln]
        current: list[str] = []
        for i, ln in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            # A heading is short, unpunctuated, and followed by a new sentence (not a wrapped line).
            is_heading = (len(ln) < 80 and not re.search(r"[.!?:;,\"'”]$", ln)
                          and (not nxt or nxt[0].isupper() or nxt[0].isdigit()))
            if is_heading and not current:
                paragraphs.append(ln)
                continue
            current.append(ln)
        if current:
            paragraphs.append(" ".join(current))
    return [p for p in paragraphs if p]


def _is_abbreviation(text_before: str) -> bool:
    """True when the final '.' belongs to an abbreviation or an initial, not a sentence end."""
    m = re.search(r"(\S+)\.$", text_before)
    if not m:
        return False
    raw = m.group(1).lstrip("(\"'“‘")
    token = raw.lower()
    if token in _ABBREVIATIONS:
        return True
    if len(raw) == 1 and raw.isupper():  # initials such as "J. Smith"
        return True
    return bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]", token))  # dotted forms such as "u.s.a."


def split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(paragraph):
        end = m.end(1)
        candidate = paragraph[start:end]
        if candidate.endswith(".") and _is_abbreviation(candidate):
            continue
        sentences.append(candidate.strip())
        start = m.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_long_sentence(sentence: str, max_duration: float, wps: float) -> list[str]:
    """Break an over-long sentence at clause punctuation, then conjunctions, into chunks that fit.

    Falls back to leaving the sentence whole: the sync engine can then cover it with more than
    one video clip, which is better than an unnatural mid-phrase cut.
    """
    if estimate_speech_duration(sentence, wps) <= max_duration:
        return [sentence]
    for splitter in (_CLAUSE, _CONJUNCTION):
        parts = [p for p in splitter.split(sentence) if p.strip()]
        if len(parts) < 2:
            continue
        chunks: list[str] = []
        current = ""
        for part in parts:
            trial = f"{current} {part}".strip()
            if current and estimate_speech_duration(trial, wps) > max_duration:
                chunks.append(current)
                current = part
            else:
                current = trial
        if current:
            chunks.append(current)
        # Avoid dangling fragments of one or two words.
        if len(chunks) > 1 and all(count_words(c) >= 3 for c in chunks):
            out: list[str] = []
            for c in chunks:  # each chunk is strictly shorter, so recursion terminates
                out.extend(split_long_sentence(c, max_duration, wps))
            return out
    return [sentence]


def analyze(script: str, words_per_second: float, max_scene_duration: float) -> list[Sentence]:
    units: list[Sentence] = []
    for p_index, paragraph in enumerate(split_paragraphs(script)):
        for sentence in split_sentences(paragraph):
            for chunk in split_long_sentence(sentence, max_scene_duration, words_per_second):
                units.append(Sentence(chunk, p_index, estimate_speech_duration(chunk, words_per_second)))
    return units
