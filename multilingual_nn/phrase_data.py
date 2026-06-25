import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


ASCII_WORD_RE = re.compile(r"[A-Za-z]")
REGEX_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
REGEX_PHRASE_BOUNDARY_RE = re.compile(r"\s*(?:,|;|:|\band\b|\bor\b|\bbut\b|\bwhile\b|\bthough\b)\s*", re.IGNORECASE)


def normalize_english_text(text: str) -> str:
    return "".join(character.lower() if "A" <= character <= "Z" else character for character in text.strip())


def should_passthrough_phrase(text: str) -> bool:
    normalized = normalize_english_text(text)
    if not normalized:
        return True
    return ASCII_WORD_RE.search(normalized) is None


@dataclass(frozen=True)
class QueryExample:
    example_id: str
    query: str
    row: dict[str, str]


class QueryCsvDataset(Dataset[QueryExample]):
    def __init__(self, csv_path: str | Path, query_column: str = "query", id_column: str = "id") -> None:
        self.csv_path = Path(csv_path)
        self.query_column = query_column
        self.id_column = id_column
        self.examples = self._load_examples()

    def _load_examples(self) -> list[QueryExample]:
        examples: list[QueryExample] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                raise ValueError("Input CSV has no header row")
            if self.query_column not in reader.fieldnames:
                raise ValueError(f"Input CSV must contain a {self.query_column!r} column")
            for index, row in enumerate(reader):
                query = row.get(self.query_column, "") or ""
                example_id = row.get(self.id_column, "") or str(index)
                examples.append(QueryExample(example_id=example_id, query=query, row=dict(row)))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> QueryExample:
        return self.examples[index]


class PhraseSegmenter:
    def __init__(
        self,
        backend: str = "spacy",
        spacy_model: str = "en_core_web_sm",
        phrase_max_tokens: int = 6,
    ) -> None:
        self.backend = backend
        self.spacy_model = spacy_model
        self.phrase_max_tokens = max(2, int(phrase_max_tokens))
        self._nlp = None
        if self.backend == "spacy":
            self._nlp = self._load_spacy_pipeline(spacy_model)
        elif self.backend != "regex":
            raise ValueError(f"Unsupported phrase segmentation backend: {backend}")

    def _load_spacy_pipeline(self, model_name: str):
        try:
            import spacy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "spaCy backend requested but 'spacy' is not installed. "
                "Install it with 'python3 -m pip install spacy' and download "
                f"'{model_name}', or use '--segment-backend regex'."
            ) from exc

        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run 'python3 -m spacy download {model_name}', or use '--segment-backend regex'."
            ) from exc

    def segment(self, query: str) -> list[str]:
        if self.backend == "spacy":
            return self._segment_with_spacy(query)
        return self._segment_with_regex(query)

    def _segment_with_regex(self, query: str) -> list[str]:
        sentences = [part.strip() for part in REGEX_SENT_BOUNDARY_RE.split(query.strip()) if part.strip()]
        phrases: list[str] = []
        for sentence in sentences or [query]:
            pieces = [part.strip() for part in REGEX_PHRASE_BOUNDARY_RE.split(sentence) if part.strip()]
            if not pieces:
                continue
            for piece in pieces:
                phrases.extend(self._split_long_piece(piece))
        return phrases or [query.strip()]

    def _split_long_piece(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.phrase_max_tokens:
            return [text.strip()]
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.phrase_max_tokens, len(words))
            chunks.append(" ".join(words[start:end]).strip())
            start = end
        return [chunk for chunk in chunks if chunk]

    def _segment_with_spacy(self, query: str) -> list[str]:
        assert self._nlp is not None
        doc = self._nlp(query)
        phrases: list[str] = []
        for sentence in doc.sents:
            sentence_phrases = self._segment_sentence_with_spacy(sentence)
            phrases.extend(sentence_phrases)
        return phrases or [query.strip()]

    def _segment_sentence_with_spacy(self, sentence) -> list[str]:
        tokens = [token for token in sentence if not token.is_space]
        if not tokens:
            return []

        noun_chunk_bounds: dict[int, int] = {}
        if sentence.doc.has_annotation("DEP"):
            try:
                for chunk in sentence.doc[sentence.start : sentence.end].noun_chunks:
                    chunk_tokens = [token for token in chunk if not token.is_space]
                    if not chunk_tokens:
                        continue
                    start = chunk_tokens[0].i
                    end = chunk_tokens[-1].i
                    noun_chunk_bounds[start] = end
            except Exception:
                noun_chunk_bounds = {}

        phrases: list[str] = []
        current_tokens: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            absolute_index = token.i
            if absolute_index in noun_chunk_bounds:
                if current_tokens:
                    phrases.append(" ".join(current_tokens).strip())
                    current_tokens = []
                end_absolute = noun_chunk_bounds[absolute_index]
                chunk_tokens: list[str] = []
                while index < len(tokens) and tokens[index].i <= end_absolute:
                    chunk_tokens.append(tokens[index].text)
                    index += 1
                if chunk_tokens:
                    phrases.append(" ".join(chunk_tokens).strip())
                continue

            current_tokens.append(token.text)
            boundary_after = False
            if token.is_punct and token.text in {".", "!", "?", ";", ":"}:
                boundary_after = True
            elif token.pos_ in {"CCONJ", "SCONJ"}:
                boundary_after = True
            elif len(current_tokens) >= self.phrase_max_tokens:
                boundary_after = True

            if boundary_after:
                phrase = " ".join(current_tokens).strip()
                if phrase:
                    phrases.append(phrase)
                current_tokens = []
            index += 1

        if current_tokens:
            phrase = " ".join(current_tokens).strip()
            if phrase:
                phrases.append(phrase)

        cleaned = [phrase for phrase in phrases if phrase and phrase != ","]
        return cleaned or [sentence.text.strip()]


class PhraseLevelBatchCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        segmenter: PhraseSegmenter,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.max_length = max_length

    def __call__(self, batch: list[QueryExample]) -> dict[str, Any]:
        encodings: list[dict[str, list[int]]] = []
        phrase_start_masks: list[list[int]] = []
        phrase_token_ids_batch: list[list[int]] = []
        kept_phrases_batch: list[list[str]] = []
        raw_queries: list[str] = []
        example_ids: list[str] = []
        rows: list[dict[str, str]] = []

        for example in batch:
            phrases = self.segmenter.segment(example.query)
            encoding = self.tokenizer(
                phrases,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            word_ids = encoding.word_ids()
            phrase_start_mask = [0] * len(encoding["input_ids"])
            phrase_token_ids = [-1] * len(encoding["input_ids"])
            kept_phrases: list[str] = []
            previous_word_id: int | None = None
            for token_index, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                phrase_token_ids[token_index] = int(word_id)
                if word_id != previous_word_id:
                    phrase_start_mask[token_index] = 1
                    kept_phrases.append(phrases[word_id])
                    previous_word_id = word_id
            encodings.append(
                {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                }
            )
            phrase_start_masks.append(phrase_start_mask)
            phrase_token_ids_batch.append(phrase_token_ids)
            kept_phrases_batch.append(kept_phrases)
            raw_queries.append(example.query)
            example_ids.append(example.example_id)
            rows.append(example.row)

        padded = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        max_seq_len = int(padded["input_ids"].shape[1])
        padded_phrase_start_masks = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        padded_phrase_token_ids = torch.full((len(batch), max_seq_len), -1, dtype=torch.long)
        for row_index, mask in enumerate(phrase_start_masks):
            padded_phrase_start_masks[row_index, : len(mask)] = torch.tensor(mask, dtype=torch.bool)
        for row_index, phrase_token_ids in enumerate(phrase_token_ids_batch):
            padded_phrase_token_ids[row_index, : len(phrase_token_ids)] = torch.tensor(phrase_token_ids, dtype=torch.long)

        phrase_counts = torch.tensor([len(phrases) for phrases in kept_phrases_batch], dtype=torch.long)
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "word_start_mask": padded_phrase_start_masks,
            "word_id_tensor": padded_phrase_token_ids,
            "phrases": kept_phrases_batch,
            "phrase_counts": phrase_counts,
            "example_ids": example_ids,
            "raw_queries": raw_queries,
            "rows": rows,
        }
