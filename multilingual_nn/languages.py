from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    code: str
    translate_code: str


ALL_LANGUAGES = [
    LanguageSpec(name="Arabic", code="arb-000", translate_code="ar"),
    LanguageSpec(name="Urdu", code="urd-000", translate_code="ur"),
    LanguageSpec(name="Bengali", code="ben-000", translate_code="bn"),
    LanguageSpec(name="Punjabi", code="pan-000", translate_code="pa"),
    LanguageSpec(name="Tamil", code="tam-000", translate_code="ta"),
    LanguageSpec(name="Telugu", code="tel-000", translate_code="te"),
    LanguageSpec(name="Kannada", code="kan-000", translate_code="kn"),
    LanguageSpec(name="Thai", code="tha-000", translate_code="th"),
    LanguageSpec(name="Vietnamese", code="vie-000", translate_code="vi"),
    LanguageSpec(name="Indonesian", code="ind-000", translate_code="id"),
    LanguageSpec(name="Javanese", code="jav-000", translate_code="jw"),
    LanguageSpec(name="Tagalog", code="tgl-000", translate_code="tl"),
    LanguageSpec(name="Khmer", code="khm-000", translate_code="km"),
    LanguageSpec(name="Burmese", code="mya-000", translate_code="my"),
    LanguageSpec(name="Nepali", code="npi-000", translate_code="ne"),
    LanguageSpec(name="Uyghur", code="uig-000", translate_code="ug"),
    LanguageSpec(name="Uzbek", code="uzn-000", translate_code="uz"),
    LanguageSpec(name="Kazakh", code="kaz-000", translate_code="kk"),
    LanguageSpec(name="Kyrgyz", code="kir-000", translate_code="ky"),
    LanguageSpec(name="Tigrinya", code="tir-000", translate_code="ti"),
]

LOW_RESOURCE_LANGUAGES = [
    LanguageSpec(name="Javanese", code="jav-000", translate_code="jw"),
    LanguageSpec(name="Khmer", code="khm-000", translate_code="km"),
    LanguageSpec(name="Burmese", code="mya-000", translate_code="my"),
    LanguageSpec(name="Uyghur", code="uig-000", translate_code="ug"),
    LanguageSpec(name="Kyrgyz", code="kir-000", translate_code="ky"),
    LanguageSpec(name="Tigrinya", code="tir-000", translate_code="ti"),
]

ACTIVE_LANGUAGE_SET = "low_resource_6"
LANGUAGES = LOW_RESOURCE_LANGUAGES
LANGUAGE_NAMES = [language.name for language in LANGUAGES]
NUM_LANGUAGES = len(LANGUAGES)
