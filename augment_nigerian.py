"""
Augment Complaint Sense training data with Nigerian language variants.

AfriSenti is sentiment analysis (positive/negative/neutral) — it cannot be
merged label-for-label with complaint categories. Instead we:

1. Keep the same Category label from your complaint CSV.
2. Create code-mixed / Pidgin variants of each complaint (phrase substitution).
3. Optionally mine complaint-style phrasing from AfriSenti Nigerian Pidgin (pcm).

Usage:
  python augment_nigerian.py
  python augment_nigerian.py --input data/complaint_sense_working.csv --variants 3
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

SEED = 42
TEXT_COL = "text"
LABEL_COL = "Category"
ID_COL = "ComplaintId"

# Discourse markers sampled per language (keeps complaint meaning, adds local tone)
OPENERS = {
    "pidgin": [
        "Abeg ",
        "Omo ",
        "Make una hear me o, ",
        "Wetin dey sup na, ",
        "I dey vex say ",
        "Na unfair o, ",
    ],
    "yoruba": [
        "Ẹ jọ̀wọ́, ",
        "Mo ní pé ",
        "E jọ, ",
        "Ó ṣeé ṣe kí ",
    ],
    "igbo": [
        "Biko, ",
        "Nna, ",
        "Kedu ihe mere na ",
        "Biko kwe ka m kwuo na ",
    ],
    "hausa": [
        "Don Allah, ",
        "Ina roko, ",
        "Wannan ba daidai ba ne, ",
    ],
}

CLOSERS = {
    "pidgin": [
        " Abeg make una help me.",
        " This one no good at all.",
        " I don tire for this matter.",
        " Make e work well abeg.",
    ],
    "yoruba": [
        " Ẹ ṣe iranlọwọ fún mi.",
        " Mo ti rẹ.",
    ],
    "igbo": [
        " Biko nyere m aka.",
        " Obi adịghị m mma.",
    ],
    "hausa": [
        " Don Allah ku taimake ni.",
        " Na gaji da wannan.",
    ],
}

# English phrase -> localized replacement (case-insensitive word boundaries)
PHRASE_MAP = {
    "pidgin": {
        r"\bhello\b": "how far",
        r"\bhi\b": "how far",
        r"\bplease\b": "abeg",
        r"\bhelp\b": "help me",
        r"\bcomplaint\b": "complaint",
        r"\bcomplain\b": "dey complain",
        r"\bcharged\b": "dem charge me",
        r"\bcharge\b": "charge",
        r"\brefund\b": "return my money",
        r"\bmoney\b": "money",
        r"\baccount\b": "account",
        r"\bbank\b": "bank",
        r"\bunauthorized\b": "without my permission",
        r"\bfraud\b": "fraud",
        r"\bdelivery\b": "delivery",
        r"\bnot received\b": "I never receive am",
        r"\bbroken\b": "don spoil",
        r"\bdefect\b": "don get fault",
        r"\bcancel\b": "cancel",
        r"\bsubscription\b": "subscription",
        r"\bsupport\b": "support",
        r"\bwrong\b": "wrong",
        r"\bissue\b": "wahala",
        r"\bproblem\b": "problem",
        r"\bstill waiting\b": "I still dey wait",
        r"\bno response\b": "dem no reply me",
    },
    "yoruba": {
        r"\bhello\b": "bawo ni",
        r"\bplease\b": "jọ̀wọ́",
        r"\bhelp\b": "ran mi lọwọ",
        r"\bmoney\b": "owó",
        r"\baccount\b": "àkọọlì",
        r"\bbank\b": "ilé ìṣúro",
        r"\bwrong\b": "àṣìṣe",
        r"\bproblem\b": "ìṣòro",
        r"\bthank you\b": "e ṣe",
        r"\bcomplaint\b": "ìkànìyàn",
    },
    "igbo": {
        r"\bhello\b": "kedu",
        r"\bplease\b": "biko",
        r"\bhelp\b": "nyere m aka",
        r"\bmoney\b": "ego",
        r"\baccount\b": "akaụntụ",
        r"\bbank\b": "ụlọ akụ",
        r"\bwrong\b": "njehie",
        r"\bproblem\b": "nsogbu",
        r"\bcomplaint\b": "mkpesa",
        r"\bthief\b": "onye oshi",
    },
    "hausa": {
        r"\bhello\b": "sannu",
        r"\bplease\b": "don Allah",
        r"\bhelp\b": "taimake ni",
        r"\bmoney\b": "kudi",
        r"\baccount\b": "asusu",
        r"\bbank\b": "banki",
        r"\bwrong\b": "kuskure",
        r"\bproblem\b": "matsala",
        r"\bcomplaint\b": "korafi",
    },
}

CATEGORY_PIDGIN_TEMPLATES = {
    "billing": "Dem dey charge me {detail} wey I no authorize.",
    "account_access": "I no fit enter my account again, {detail}",
    "delivery_shipping": "My package never arrive, {detail}",
    "refund_return": "Abeg return my money, {detail}",
    "fraud_unauthorized": "Somebody use my account without permission, {detail}",
    "product_defect": "The thing don spoil, {detail}",
    "warranty_repair": "Repair never happen, {detail}",
    "subscription_cancel": "I cancel am but dem still dey charge me, {detail}",
    "general_inquiry": "I wan ask about {detail}",
    "customer_service": "Customer service no dey respond, {detail}",
}


def apply_phrase_map(text: str, lang: str, rng: random.Random) -> str:
    out = text
    mapping = PHRASE_MAP.get(lang, {})
    for pattern, repl in mapping.items():
        if rng.random() < 0.65:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def shorten_for_template(text: str, max_words: int = 12) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]).rstrip(".") + "..."


def augment_row(
    row: pd.Series,
    lang: str,
    rng: random.Random,
    use_template: bool,
) -> dict:
    text = str(row[TEXT_COL])
    category = row[LABEL_COL]

    if use_template and lang == "pidgin" and category in CATEGORY_PIDGIN_TEMPLATES:
        body = shorten_for_template(text)
        augmented = CATEGORY_PIDGIN_TEMPLATES[category].format(detail=body)
    else:
        augmented = apply_phrase_map(text, lang, rng)
        if rng.random() < 0.7:
            augmented = rng.choice(OPENERS.get(lang, [""])) + augmented
        if rng.random() < 0.5:
            augmented = augmented + rng.choice(CLOSERS.get(lang, [""]))

    augmented = re.sub(r"\s+", " ", augmented).strip()
    return {
        ID_COL: f"{row[ID_COL]}_{lang}",
        TEXT_COL: augmented,
        LABEL_COL: category,
        "FamilyId": f"aug:{lang}:{row.get('FamilyId', 'unknown')}",
        "augmentation": lang,
        "source_id": row[ID_COL],
    }


def load_afrisenti_pcm_complaint_phrases(afrisenti_dir: Path, max_phrases: int = 200) -> list[str]:
    """Mine short complaint-style lines from AfriSenti Pidgin split (unlabeled)."""
    pcm_path = afrisenti_dir / "data" / "pcm" / "train.tsv"
    if not pcm_path.exists():
        return []

    keyword_pattern = (
        r"\b(?:complain|charge|refund|bank|account|subscription|payment|"
        r"unfair|money|decoder|transaction|fraud|scam|service)\b"
    )
    df = pd.read_csv(pcm_path, sep="\t")
    hits = df[df["tweet"].astype(str).str.contains(keyword_pattern, case=False, na=False, regex=True)]
    phrases = hits["tweet"].astype(str).tolist()
    return phrases[:max_phrases]


def augment_dataframe(
    df: pd.DataFrame,
    variants: list[str],
    rng: random.Random,
    template_ratio: float = 0.35,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        base = row.to_dict()
        base["augmentation"] = "original"
        base["source_id"] = row[ID_COL]
        rows.append(base)

        for lang in variants:
            use_template = rng.random() < template_ratio
            rows.append(augment_row(row, lang, rng, use_template))

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=[TEXT_COL], keep="first").reset_index(drop=True)
    return out


def build_nigerian_test_set(categories: list[str]) -> pd.DataFrame:
    """Small hand-crafted Nigerian-style test complaints (one per category)."""
    samples = [
        ("billing", "Abeg why dem dey charge me every month for premium support wey I no subscribe?"),
        ("account_access", "Mo ni pe account mi ti di locked, mo ko le wọle mọ."),
        ("delivery_shipping", "I never receive my package o, e don pass two weeks."),
        ("refund_return", "Biko return my money, dem give me wrong product."),
        ("fraud_unauthorized", "Somebody use my card without permission, barawo dem!"),
        ("product_defect", "The laptop don spoil after two weeks, na wahala."),
        ("warranty_repair", "Dem cancel repair appointment twice, I don tire."),
        ("subscription_cancel", "I cancel subscription but dem still dey charge me kudi."),
        ("general_inquiry", "Don Allah, una get student pricing for annual license?"),
        ("customer_service", "Customer service no dey reply, wetin dey sup na?"),
    ]
    rows = []
    for i, (cat, text) in enumerate(samples):
        if cat in categories:
            rows.append(
                {
                    ID_COL: f"nigerian_test_{i}",
                    TEXT_COL: text,
                    LABEL_COL: cat,
                    "FamilyId": "nigerian_test",
                    "augmentation": "manual_test",
                    "source_id": None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment complaints with Nigerian languages")
    parser.add_argument("--input", type=Path, default=Path("data/complaint_sense_working.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/complaint_sense_augmented.csv"))
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/nigerian_test_complaints.csv"),
    )
    parser.add_argument(
        "--afrisenti-dir",
        type=Path,
        default=Path("afrisenti"),
        help="Local AfriSenti clone (optional, for phrase mining stats)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pidgin", "yoruba", "igbo", "hausa"],
        choices=["pidgin", "yoruba", "igbo", "hausa"],
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.input)
    categories = sorted(df[LABEL_COL].unique().tolist())

    augmented = augment_dataframe(df, args.variants, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False)

    test_df = build_nigerian_test_set(categories)
    test_df.to_csv(args.test_output, index=False)

    pcm_phrases = load_afrisenti_pcm_complaint_phrases(args.afrisenti_dir)
    print("=== Nigerian augmentation summary ===")
    print(f"Input rows:     {len(df)}")
    print(f"Output rows:    {len(augmented)}")
    print(f"By augmentation:\n{augmented['augmentation'].value_counts().to_string()}")
    print(f"\nWrote training: {args.output}")
    print(f"Wrote test set: {args.test_output} ({len(test_df)} rows)")
    print(f"AfriSenti pcm complaint-style phrases mined: {len(pcm_phrases)}")
    if pcm_phrases:
        print("Sample mined phrase:", pcm_phrases[0][:120])


if __name__ == "__main__":
    main()
