"""
כלים משותפים: נרמול טקסט עברי, ומילון מונחים טכניים (glossary).

פורמט glossary.txt, שורה לכל מונח:
    deployment | דיפלוימנט | דפלוימנט
    Jira | ג'ירה | גירה
המילה הראשונה היא הצורה "הרשמית". כל הצורות בשורה נחשבות תמלול נכון של המונח.
"""
import re
from pathlib import Path

NIQQUD = re.compile(r"[֑-ׇ]")
NON_WORD = re.compile(r"[^\w\s]")


def normalize_he(text: str) -> str:
    """נרמול לפני השוואה, כדי שפיסוק, ניקוד ואותיות גדולות לא ייספרו כטעויות."""
    text = text.replace("־", " ")        # מקף עברי -> רווח
    text = NIQQUD.sub("", text)
    # גרש וגרשיים נמחקים, כך ש"צה״ל"="צהל" ו"ג'ירה"="גירה"
    text = re.sub(r"[\"'״׳`]", "", text)
    text = text.replace("-", " ")
    text = NON_WORD.sub(" ", text)
    return " ".join(text.lower().split())


def load_glossary(path) -> list[list[str]]:
    """מחזיר רשימה של מונחים, כל מונח הוא רשימת צורות מנורמלות. הראשונה היא הרשמית."""
    p = Path(path)
    if not p.exists():
        return []
    terms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        variants = [normalize_he(v) for v in line.split("|") if v.strip()]
        if variants:
            terms.append(variants)
    return terms


def glossary_prompt(path, max_terms: int = 60) -> str | None:
    """
    טקסט שמוזן ל-Whisper כ-initial_prompt, כדי "להכין" אותו למונחים.
    Whisper מתחשב רק בכ-200 טוקנים של prompt, לכן מגבילים את מספר המונחים.
    """
    p = Path(path)
    if not p.exists():
        return None
    canon = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            canon.append(line.split("|")[0].strip())
    if not canon:
        return None
    return "פגישת צוות פיתוח. מונחים: " + ", ".join(canon[:max_terms]) + "."


HEB_PREFIX = "(?:[ובכלמשה]{1,3})?"  # אותיות שימוש: "הדיפלוימנט", "בג'ירה", "שנמרג'ג'"


def _contains(text_n: str, phrase_n: str) -> bool:
    """המונח מופיע כמילה שלמה, עם אפשרות לאותיות שימוש בעברית לפניו."""
    pattern = rf"(?:^|\s){HEB_PREFIX}{re.escape(phrase_n)}(?:\s|$)"
    return re.search(pattern, text_n) is not None


def term_hits(reference: str, hypothesis: str, terms: list[list[str]]) -> tuple[int, int, list[str]]:
    """
    Term Recall: מתוך המונחים שבאמת נאמרו (מופיעים בתמלול הנכון),
    כמה המודל תמלל נכון, בכל אחת מהצורות המותרות.
    מחזיר (נאמרו, תומללו נכון, רשימת מונחים שפוספסו).
    """
    ref_n, hyp_n = normalize_he(reference), normalize_he(hypothesis)
    said = hit = 0
    missed = []
    for variants in terms:
        if any(_contains(ref_n, v) for v in variants):
            said += 1
            if any(_contains(hyp_n, v) for v in variants):
                hit += 1
            else:
                missed.append(variants[0])
    return said, hit, missed
