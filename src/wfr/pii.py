"""PII handling. The LLM only ever sees employee IDs and aggregates, never names/emails/phones."""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\- ]{8,}\d")


def mask_text(text: str, known_names: list[str] | None = None) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    for n in known_names or []:
        text = re.sub(re.escape(n), "[NAME]", text, flags=re.IGNORECASE)
    return text


def find_pii(text: str, known_names: list[str] | None = None) -> list[str]:
    hits = EMAIL.findall(text) + PHONE.findall(text)
    for n in known_names or []:
        if n.lower() in text.lower():
            hits.append(n)
    return hits


DIRECT_IDENTIFIERS = ["name", "email", "phone"]


def strip_direct_identifiers(df):
    return df.drop(columns=[c for c in DIRECT_IDENTIFIERS if c in df.columns])
