# Complaint Sense — Consumer Complaint Classification

Challenge 9: Intelligent Complaint Classification Using Localized Transformer Architectures (Team Stanley).

Fine-tune **DistilBERT** on CFPB consumer complaint narratives to predict one of **11 Issue** categories. Primary metric: **Macro F1**.

## Project layout

| Path | Purpose |
|------|---------|
| `complaint_sense_distilbert.ipynb` | End-to-end notebook (load → preprocess → train → evaluate) |
| `prepare_data.py` | Build the balanced working CSV from a raw CFPB dump |
| `requirements.txt` | Python dependencies |
| `data/` | Place `complaint_sense_working.csv` or `complaints.csv` here |
| `models/` | Saved fine-tuned model |
| `outputs/` | Metrics, plots, train/val/test splits |

## Quick start

```bash
pip install -r requirements.txt

# Option A — you already exported a filtered CFPB CSV with narratives
# Save it as data/complaints.csv, then:
python prepare_data.py --input data/complaints.csv --per-class 30

# Option B — download the full public CFPB dump (large), then filter:
python prepare_data.py --download --per-class 30

# Open and run the notebook
# complaint_sense_distilbert.ipynb
```

If you already have the data-card working subset (≈323 rows), save it as:

`data/complaint_sense_working.csv`

with at least columns `Consumer complaint narrative` and `Issue`.

## Target labels (11)

1. Incorrect information on your report  
2. Improper use of your report  
3. Managing an account  
4. Problem with a company's investigation into an existing problem  
5. Problem with a purchase shown on your statement  
6. Attempts to collect debt not owed  
7. Trouble during payment process  
8. False statements or representation  
9. Struggling to pay mortgage  
10. Written notification about debt  
11. Took or threatened to take negative or legal action  

## Data source

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — CC0 1.0 / U.S. public domain (subject to CFPB stated exceptions). Narratives appear only when consumers opt in; identifying information is redacted by CFPB.

## Nigerian languages (Yoruba, Igbo, Hausa, Pidgin)

**AfriSenti cannot be merged directly** with complaint labels — it is sentiment analysis (positive/negative/neutral), not complaint routing.

Instead, use AfriSenti as a **language/style resource**:

1. Run augmentation (creates Pidgin + code-mixed variants of your complaints, same `Category` label):
   ```bash
   python augment_nigerian.py
   ```
2. In the notebook, set `USE_NIGERIAN_AUGMENTATION = True` and re-train.
3. Evaluate on `data/nigerian_test_complaints.csv` (section 7b in the notebook).

AfriSenti folders used: `afrisenti/data/pcm` (Nigerian Pidgin), plus `yor`, `ibo`, `hau` for phrasing patterns.

**Limitation:** Augmented data is synthetic code-mixing, not real labeled Nigerian complaints. For production, collect verified Nigerian complaint narratives with Issue/Category labels.

## Note on narratives

As of mid-August 2026, CFPB announced it would cease discretionary publication of new complaint narratives. Historical narrative-bearing exports (including any extract your team already downloaded for the data card) remain the intended training source for this project.
