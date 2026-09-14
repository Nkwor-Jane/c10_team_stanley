"""
Augment Complaint Sense training data with Nigerian language variants.

AfriSenti is sentiment analysis (positive/negative/neutral) — it cannot be
merged label-for-label with complaint categories. Instead we:

1. Keep the same Issue label from your complaint CSV.
2. Create code-mixed / Pidgin variants of each complaint (phrase substitution).
3. Optionally mine complaint-style phrasing from AfriSenti Nigerian Pidgin (pcm).

Usage:
  python augment_nigerian.py
  python augment_nigerian.py --input data/complaint_sense_working.csv --variants pidgin yoruba
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

SEED = 42

# Must match the CFPB export / Data Card schema used by prepare_data.py
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Issue"
ID_COL = "Complaint ID"

TARGET_ISSUES = [
    "Incorrect information on your report",
    "Improper use of your report",
    "Managing an account",
    "Problem with a company's investigation into an existing problem",
    "Problem with a purchase shown on your statement",
    "Attempts to collect debt not owed",
    "Trouble during payment process",
    "False statements or representation",
    "Struggling to pay mortgage",
    "Written notification about debt",
    "Took or threatened to take negative or legal action",
]


def resolve_root() -> Path:
    """
    Resolve the project root, whether this script is run from Tri-ai/ or
    Tri-ai/scripts/, so data/ is always read from and written to the same
    sibling folder as scripts/ — matching prepare_data.py and the notebook.
    """
    root = Path(".").resolve()
    if root.name == "scripts":
        root = root.parent
    return root


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
        r"\bdebt\b": "debt",
        r"\bmortgage\b": "house loan",
        r"\breport\b": "report",
        r"\bstatement\b": "statement",
        r"\bpayment\b": "payment",
        r"\bnegative\b": "bad",
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
        r"\bdebt\b": "gbèsè",
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
        r"\bdebt\b": "ụgwọ",
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
        r"\bdebt\b": "bashi",
    },
}

# Short Pidgin templates keyed on the real CFPB Issue categories
CATEGORY_PIDGIN_TEMPLATES = {
    "Incorrect information on your report": "Dem put wrong info for my report, {detail}",
    "Improper use of your report": "Somebody use my report anyhow, {detail}",
    "Managing an account": "I dey struggle to manage my account, {detail}",
    "Problem with a company's investigation into an existing problem": (
        "Dem no investigate my matter well, {detail}"
    ),
    "Problem with a purchase shown on your statement": "I no recognize this purchase for my statement, {detail}",
    "Attempts to collect debt not owed": "Dem dey try collect debt wey I no owe, {detail}",
    "Trouble during payment process": "I dey get wahala when I wan pay, {detail}",
    "False statements or representation": "Dem tell me lie about this matter, {detail}",
    "Struggling to pay mortgage": "I dey find am hard to pay my house loan, {detail}",
    "Written notification about debt": "Dem send me letter about debt wey no correct, {detail}",
    "Took or threatened to take negative or legal action": "Dem threaten to take legal action against me, {detail}",
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
        r"unfair|money|decoder|transaction|fraud|scam|service|debt|mortgage)\b"
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
    """Small hand-crafted Nigerian-style test complaints, one per real Issue category."""
    samples = [
        ("Incorrect information on your report", "Abeg my credit report get wrong info wey I never do."),
        ("Improper use of your report", "Somebody use my report anyhow without my permission."),
        ("Managing an account", "Mo ni pe account mi ti di locked, mo ko le wọle mọ."),
        (
            "Problem with a company's investigation into an existing problem",
            "Dem no investigate my matter well, na wahala be dis.",
        ),
        (
            "Problem with a purchase shown on your statement",
            "I no recognize this purchase for my statement o.",
        ),
        ("Attempts to collect debt not owed", "Dem dey try collect debt wey I no owe, na lie."),
        ("Trouble during payment process", "I dey get wahala anytime I wan pay, e no dey work."),
        ("False statements or representation", "Dem tell me lie about this matter, e no correct."),
        ("Struggling to pay mortgage", "I dey find am hard to pay my house loan every month."),
        ("Written notification about debt", "Dem send me letter about debt wey no correct at all."),
        (
            "Took or threatened to take negative or legal action",
            "Dem threaten to take legal action against me, barawo dem!",
        ),
    ]
    rows = []
    for i, (cat, text) in enumerate(samples):
        if cat in categories:
            rows.append(
                {
                    ID_COL: f"nigerian_test_{i}",
                    TEXT_COL: text,
                    LABEL_COL: cat,
                    "augmentation": "manual_test",
                    "source_id": None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    root = resolve_root()
    data_dir = root / "data"

    parser = argparse.ArgumentParser(description="Augment complaints with Nigerian languages")
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "complaint_sense_working.csv",
        help="Defaults to <project_root>/data/complaint_sense_working.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / "complaint_sense_augmented.csv",
        help="Defaults to <project_root>/data/complaint_sense_augmented.csv",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=data_dir / "nigerian_test_complaints.csv",
        help="Defaults to <project_root>/data/nigerian_test_complaints.csv",
    )
    parser.add_argument(
        "--afrisenti-dir",
        type=Path,
        default=root / "afrisenti",
        help="Local AfriSenti clone (optional, for phrase mining stats). "
        "Defaults to <project_root>/afrisenti.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pidgin", "yoruba", "igbo", "hausa"],
        choices=["pidgin", "yoruba", "igbo", "hausa"],
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found. Run prepare_data.py first to build the working "
            f"dataset, or pass --input pointing at your complaint CSV."
        )

    rng = random.Random(args.seed)
    df = pd.read_csv(args.input)

    missing = [c for c in (ID_COL, TEXT_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{args.input} is missing required columns {missing}.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected columns: {[ID_COL, TEXT_COL, LABEL_COL]} "
            f"(matching the CFPB export / Data Card schema)."
        )

    categories = sorted(df[LABEL_COL].unique().tolist())

    augmented = augment_dataframe(df, args.variants, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False)

    test_df = build_nigerian_test_set(categories)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
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