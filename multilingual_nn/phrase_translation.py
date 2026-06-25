import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .languages import LANGUAGES, LanguageSpec
from .phrase_data import normalize_english_text, should_passthrough_phrase


TRANSLATE_API_URL = "https://translate.googleapis.com/translate_a/single"


@dataclass(frozen=True)
class PhraseTranslation:
    source_phrase: str
    normalized_phrase: str
    translated_phrase: str
    language_name: str
    language_code: str
    status: str


@dataclass(frozen=True)
class PhraseTranslationResult:
    translated_phrases: list[str]
    translated_query: str
    phrase_translations: list[PhraseTranslation]
    fail_rate: float
    expansion_rate: float


class GoogleTranslatePhraseTranslator:
    def __init__(
        self,
        source_language: str = "en",
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        language_specs: list[LanguageSpec] | None = None,
    ) -> None:
        self.source_language = source_language
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.language_specs = list(language_specs) if language_specs is not None else list(LANGUAGES)
        self.cache: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._cache_lock = threading.Lock()

    def translate_batch(
        self,
        batch_phrases: list[list[str]],
        batch_actions: list[list[int]],
    ) -> list[PhraseTranslationResult]:
        results: list[PhraseTranslationResult] = []
        for phrases, actions in zip(batch_phrases, batch_actions):
            results.append(self.translate_phrases(phrases, actions))
        return results

    def _resolve_language(self, action: int) -> LanguageSpec:
        index = int(action)
        if index < 0 or index >= len(self.language_specs):
            raise IndexError(f"Language action index out of range: {index} for {len(self.language_specs)} languages.")
        return self.language_specs[index]

    def prefetch_batch(
        self,
        batch_phrases: list[list[str]],
        batch_actions: list[list[int]],
        max_workers: int = 1,
    ) -> None:
        pending: dict[tuple[str, str, str], LanguageSpec] = {}
        for phrases, actions in zip(batch_phrases, batch_actions):
            for phrase, action in zip(phrases, actions):
                language = self._resolve_language(action)
                normalized_phrase = normalize_english_text(phrase)
                cache_key = (normalized_phrase, self.source_language, language.translate_code)
                if should_passthrough_phrase(normalized_phrase):
                    with self._cache_lock:
                        self.cache.setdefault(cache_key, (phrase, "passthrough"))
                    continue
                with self._cache_lock:
                    if cache_key in self.cache:
                        continue
                pending[cache_key] = language

        if not pending:
            return

        worker_count = max(1, min(int(max_workers), len(pending)))
        if worker_count == 1:
            for cache_key, language in pending.items():
                translated_phrase, status = self._fetch_translation(cache_key[0], language)
                with self._cache_lock:
                    self.cache[cache_key] = (translated_phrase, status)
            return

        def _worker(item: tuple[tuple[str, str, str], LanguageSpec]) -> tuple[tuple[str, str, str], tuple[str, str]]:
            cache_key, language = item
            translated_phrase, status = self._fetch_translation(cache_key[0], language)
            return cache_key, (translated_phrase, status)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for cache_key, value in executor.map(_worker, pending.items()):
                with self._cache_lock:
                    self.cache[cache_key] = value

    def translate_phrases(self, phrases: list[str], actions: list[int]) -> PhraseTranslationResult:
        translated_phrases: list[str] = []
        phrase_translations: list[PhraseTranslation] = []
        fallback_count = 0
        translated_word_count = 0
        original_word_count = sum(len(phrase.split()) for phrase in phrases)

        for phrase, action in zip(phrases, actions):
            language = self._resolve_language(action)
            translated_phrase, status, normalized_phrase = self.translate_phrase(phrase, language)
            translated_phrases.append(translated_phrase)
            translated_word_count += len(translated_phrase.split()) if translated_phrase else 0
            if status == "fallback":
                fallback_count += 1
            phrase_translations.append(
                PhraseTranslation(
                    source_phrase=phrase,
                    normalized_phrase=normalized_phrase,
                    translated_phrase=translated_phrase,
                    language_name=language.name,
                    language_code=language.code,
                    status=status,
                )
            )

        phrase_count = max(len(phrases), 1)
        translated_query = " ".join(phrase.strip() for phrase in translated_phrases if phrase.strip())
        fail_rate = fallback_count / phrase_count
        expansion_rate = max(0, translated_word_count - max(original_word_count, 1)) / max(original_word_count, 1)
        return PhraseTranslationResult(
            translated_phrases=translated_phrases,
            translated_query=translated_query,
            phrase_translations=phrase_translations,
            fail_rate=fail_rate,
            expansion_rate=expansion_rate,
        )

    def translate_phrase(self, phrase: str, language: LanguageSpec) -> tuple[str, str, str]:
        normalized_phrase = normalize_english_text(phrase)
        cache_key = (normalized_phrase, self.source_language, language.translate_code)
        with self._cache_lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            translated_phrase, status = cached
            return translated_phrase, status, normalized_phrase

        if should_passthrough_phrase(normalized_phrase):
            with self._cache_lock:
                self.cache[cache_key] = (phrase, "passthrough")
            return phrase, "passthrough", normalized_phrase

        translated_phrase, status = self._fetch_translation(normalized_phrase, language)
        with self._cache_lock:
            self.cache[cache_key] = (translated_phrase, status)
        return translated_phrase, status, normalized_phrase

    def _fetch_translation(self, normalized_phrase: str, language: LanguageSpec) -> tuple[str, str]:
        if should_passthrough_phrase(normalized_phrase):
            return normalized_phrase, "passthrough"

        params = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": self.source_language,
                "tl": language.translate_code,
                "dt": "t",
                "q": normalized_phrase,
            }
        )
        request = urllib.request.Request(
            f"{TRANSLATE_API_URL}?{params}",
            headers={"User-Agent": "Mozilla/5.0"},
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                translated = "".join(part[0] for part in payload[0]).strip()
                if not translated:
                    raise ValueError(f"Empty translation result for phrase {normalized_phrase!r}")
                return translated, "api"
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
                delay_seconds = min(60.0, self.backoff_seconds * max(1, attempt))
                print(
                    f"\n[wait] translation retryable error on attempt {attempt}; "
                    f"sleeping {delay_seconds:.1f}s: {type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(delay_seconds)
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"Translation failed with non-retryable local error: {error}") from error
