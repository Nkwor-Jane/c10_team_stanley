# Complaint Sense — Consumer Complaint Classification

Challenge 9: Intelligent Complaint Classification Using Localized Transformer Architectures (Team Stanley). Fine-tunes **DistilBERT** to predict a complaint's **Issue category** from its free-text narrative. Primary metric: **Macro F1**.

## Dataset

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (CC0 1.0 public-domain dedication, subject to CFPB's stated exceptions). Narratives are published only when consumers opt in, and CFPB redacts identifying information before release.

**Selection process:** 1,293 complaints with published narratives were exported (period 2025-08-16 to 2026-08-16). After removing exact-duplicate narratives, 1,166 unique complaints remained. From these, 11 CFPB Issue categories were selected — prioritizing categories with clear meaning and sufficient volume, and excluding the vague "Other features, terms, or problems" and "Dealing with your lender or servicer" labels. The current balanced working subset caps each class at 30 examples (23 for the smallest class), for 323 total complaints.

**Localization extension:** The CFPB set is U.S., English-only, and financial-services-specific. To probe generalization toward Nigerian markets, we use **AfriSenti** (a sentiment-labeled African-language corpus) as a *style/phrasing resource* only, not a label source — sentiment labels don't map to Issue categories. `augment_nigerian.py` uses AfriSenti's Pidgin (`pcm`), Yoruba, Igbo, and Hausa subsets to generate Nigerian Pidgin/code-mixed variants of existing complaints, keeping each original Issue label. A small `nigerian_test_complaints.csv` holds out Nigerian-style narratives for evaluation. **Limitation:** augmented data is synthetic code-mixing, not verified real-world Nigerian complaints.

## Training Pipeline

1. **Collection** (`prepare_data.py`): reads a raw CFPB export, filters to narratives longer than 20 characters, drops empty/null text, keeps only the 11 target Issue categories, and caps each class at `--per-class` examples (default 30).
2. **Preprocessing**: whitespace normalization only — punctuation, slang, and typos are preserved deliberately so the model sees realistic informal language. Exact-duplicate narratives are dropped before splitting to prevent leakage.
3. **Splitting**: stratified 80/20 train/test split, then a further stratified 90/10 train/validation split (both stratified on label), seed = 42.
4. **Augmentation (train only)**: if enabled, Nigerian-language variants generated from `augment_nigerian.py` are appended to the train split *after* the split is fixed, so validation/test stay free of augmented or leaked rows.
5. **Tokenization**: `distilbert-base-uncased` tokenizer, max sequence length 256, padded/truncated.
6. **Model & hyperparameters**: `AutoModelForSequenceClassification` on `distilbert-base-uncased`, 11-way head. Settled on: LR 2e-5, weight decay 0.01, warmup ratio 0.1, 4 epochs, batch size 8 train / 16 eval on GPU (4/8 on CPU), fp16 on GPU, early stopping (patience 2) and best-checkpoint selection both driven by validation **macro F1**.
7. **Design choices**: macro F1 (not accuracy) drives selection so minority categories aren't ignored; light cleaning and a fixed seed (42) keep results reproducible.

## Evaluation

- **Held-out English test set** (20%, stratified, de-duplicated beforehand): accuracy, macro F1 (primary), weighted F1, macro precision/recall, full per-class precision/recall/F1 report, and a confusion matrix (`outputs/confusion_matrix.png`, `outputs/per_class_f1.png`).
- **Nigerian-language test set**: the same trained model is scored separately on `nigerian_test_complaints.csv` (Pidgin/code-mixed narratives) to check whether performance holds up on localized language it wasn't directly trained on — reported as its own accuracy/macro F1 (`outputs/nigerian_test_metrics.json`), not blended into the main test score.
- **Leakage checks**: exact-duplicate narratives removed pre-split; augmentation added only to the train split.
- All metrics and split files (`train.csv`, `val.csv`, `test.csv`, `test_metrics.json`, `per_class_metrics.csv`) are written to `outputs/` for auditability.
- An inference helper (`predict_complaint`) returns top-k predicted categories with probabilities for spot-checking single complaints.

**Known limitation:** the working subset (323 rows) is small; treat current metrics as a baseline, not a final result. Ambiguous or high-risk complaints (fraud, legal action, safety) should still be routed to human review rather than decided automatically.

## Reproduction

Run in this order:

```bash
pip install -r requirements.txt

# 1. Build the balanced working CSV
#    Option A: you already have a filtered CFPB CSV with narratives
python prepare_data.py --input data/complaints.csv --per-class 30
#    Option B: download the full public CFPB dump, then filter
python prepare_data.py --download --per-class 30

# 2. (Optional) generate Nigerian-language augmented variants
python augment_nigerian.py

# 3. Open and run the notebook top to bottom
#    complaint_sense_distilbert.ipynb
#    (load -> exploratory checks -> split/tokenize -> fine-tune -> evaluate)
```

If you already have the Data Card's balanced subset (~323 rows), save it directly as `data/complaint_sense_working.csv` with at least `Consumer complaint narrative` and `Issue` columns; step 1 can then be skipped. Set `USE_NIGERIAN_AUGMENTATION = True/False` in the notebook's configuration cell to toggle step 2's effect before training.

**Note on narratives:** as of mid-August 2026, CFPB announced it would stop discretionary publication of new complaint narratives. Historical narrative-bearing exports (including the one used for this project's Data Card) remain the intended training source going forward.

## Appendix

**Team Stanley**
- Nkwor Jane Chinelo
- Oke Oluwajomiloju

**Mentors**
- David Brown Balogun

## References
- CFPB Consumer Complaint Database — https://www.consumerfinance.gov/data-research/consumer-complaints/
- AfriSenti (African-language sentiment corpus), used as a style/phrasing resource for Nigerian-language augmentation - # Complaint Sense — Consumer Complaint Classification