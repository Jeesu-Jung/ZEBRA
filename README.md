# ZEBRA — ZEro‑annotation **B**Ehavior‑based **R**esponse **A**lignment  
_Comprehensive README_

---

## 1. Project Scope

**ZEBRA** provides a zero‑annotation pipeline for constructing large‑scale **preference datasets** for LLM alignment **without any per‑instance labeling**.  
It binarizes response pairs by leveraging **model‑behavior knowledge**—public benchmark scores—instead of human or GPT‑4 judgments.  
All details are described in the accompanying EMNLP‑2025 paper.

---

## 2. Repository Layout

```
.
├── _EMNLP_2025__ZEBRA.pdf          # The paper
├── Benchmark_scores.xlsx           # Raw benchmark table (optional helper)
├── binarize.score.py              # Quality‑first (Aq) constructor
├── binarize.similarity.py               # Similarity‑first (As) constructor
├── binarize.combined.py  # Hybrid (Aqs) constructor
└── model_aware/
    └── benchmark.csv               # Normalised benchmark table used by the scripts
```

| Script | Anchoring Strategy | Description |
|--------|-------------------|-------------|
| `binarize.score.py` | **Aq** | Ranks models by mean benchmark score and pairs **top‑1 vs top‑2** to build preference data. |
| `binarize.similarity.py` | **As** | Finds the **most behaviour‑similar** model pair (cos ≥ 0.9) per instruction and labels by the higher‑ranked model. |
| `binarize.combined.py` | **Aqs** | Blends quality and similarity with tunable weights (`rank_weight`, `similarity_weight`). |

All three emit a JSON file that mirrors the UltraFeedback schema, ready for SFT or DPO.

---

## 3. Installation

```bash
# Python ≥3.10 is recommended
python -m venv zebra-env && source zebra-env/bin/activate
pip install --upgrade pip

# Core scientific stack
pip install numpy pandas scikit-learn matplotlib tqdm
# HF datasets for UltraFeedback
pip install datasets
```

---

## 4. Data Prerequisites

1. **Benchmark matrix** – provided as `benchmark.csv` (17 models × 6 tasks).  
   *Tasks:* IFeval, MMLU‑STEM, MMLU‑Pro, Hellaswag, ARC‑Easy, ARC‑Challenge.  
2. **UltraFeedback** – downloaded on the fly by the scripts:

   ```python
   from datasets import load_dataset
   ds = load_dataset("openbmb/UltraFeedback")
   ```

---

## 5. Running the Preference Constructors

```bash
# Quality‑first
python binarize.score.py

# Similarity‑first
python binarize.similarity.py

# Hybrid (adjust weights inside the script if desired)
python binarize.combined.py
```

### Output (default)

```
model_aware/UltraFeedback.similarity.anchor.json          # Aq
model_aware/UltraFeedback.least.score_ranking.no_ultra65.json   # As
model_aware/UltraFeedback.combined.json                   # Aqs
```

Each record follows the UltraFeedback structure, with additional fields:

```json
{
  "prompt": "...",
  "most_similar_pair": ["model_A", "model_B"],
  "chosen":   [...],
  "rejected": [...],
  "kinship_rank": 1
}
```

---

## 6. Alignment‑Tuning Recipe (SFT / DPO)

The paper fine‑tunes **Llama‑3.1** (3 B & 8 B) and **Qwen‑2.5** (3 B & 8 B).

| batch/device | grad‑acc | lr | warm‑up | precision |
|--------------|----------|----|---------|-----------|
| 6 | 4 | 5 × 10⁻⁵ | 500 steps | bf16 |

Example using HuggingFace `trl` **DPOTrainer**:

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model,
    ref_model,
    train_dataset,             # JSON from Aqs or other strategy
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True,
)
trainer.train()
```

Evaluation uses the same six benchmarks (scripts are provided in the artifact directory).

---

## 7. Reproducing the Paper’s Numbers

1. Generate preference JSONs with the three constructor scripts.  
2. Fine‑tune your base model with SFT _or_ DPO on each JSON file.  
3. Evaluate the resulting checkpoints with the provided harness—average scores should match Table 2 (±0.01).

---

## 8. Extending ZEBRA

* **Add new models** – append their benchmark scores to `benchmark.csv`, then rerun the constructor scripts.  
* **Adjust similarity threshold** – modify the `threshold` variable (default = 0.9 cosine).  
* **Curriculum tuning** – feed Aq → As → Aqs data sequentially to simulate coarse‑to‑fine alignment (see §5.2 of the paper).

---

## 10. License & Contact

All code is released under the MIT License unless noted otherwise.  
Questions or issues?  Open an issue here or contact the corresponding author listed in the paper.

---

**Enjoy cost‑free preference construction with ZEBRA!**
