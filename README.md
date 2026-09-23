# Kazakhstan History QA Bot

Fine-tuning a QA model on the 11th-grade Kazakh history textbook (pages 3–23, scanned).

**Pipeline:** PDF (scans) → Docling/EasyOCR → GPT-5-mini teacher → embedding dedup → retrieval-grounding filter → Unsloth QLoRA SFT → GPT-5 judge eval → HuggingFace Hub

**Fine-tuned adapter:** [zhadyrazhan/kz-history-qwen2.5-3b-lora](https://huggingface.co/zhadyrazhan/kz-history-qwen2.5-3b-lora)

---

## Results

| Stage | Result |
|---|---|
| OCR | 29,311 characters, 39 chunks |
| SFT dataset | **154 pairs** (`data/history_sft_dataset.json`) |
| Training | 160 steps, 8 epochs, loss **2.67 → 0.14** |
| LLM-as-judge | accuracy **2.60 / 5**, faithfulness **3.20 / 5** |

### Three distinct models

No model ever grades its own output:

| Role | Model |
|---|---|
| Teacher (QA generation) | `gpt-5-mini` |
| Student (fine-tuned) | `Qwen2.5-3B-Instruct` (4-bit) |
| Judge (evaluation) | `gpt-5` |

---

## Training statistics

**Hyperparameters**

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| LoRA rank / alpha | 16 / 16 |
| Trainable parameters | 29,933,568 of 3,115,872,256 (**0.96%**) |
| Batch size (effective) | 8 (2 × grad accum 4) |
| Learning rate | 2e-4 |
| Epochs / steps | 8 / 160 |
| Max seq length | 2048 |

**Loss reduction** (full plot in `data/loss_curve.png`, notebook section 11)

| Step | 10 | 30 | 60 | 90 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|---|
| Loss | 2.674 | 0.848 | 0.662 | 0.411 | 0.301 | 0.185 | **0.140** |

**Dataset composition** — all four question types required by the brief:

| Type | Count |
|---|---|
| `кто_такой` (who is) | 46 |
| `значение` (significance) | 46 |
| `век` (which century) | 31 |
| `что_написал` (what did they write) | 31 |

---

## What actually moved accuracy

Four runs. Important caveat: more than one variable changed between runs, so this is an experiment log, not a controlled study.

| # | Pairs | Epochs | Accuracy | Faithfulness | What changed |
|---|---|---|---|---|---|
| 1 | 80 | 3 | 1.00 | 1.20 | baseline run |
| 2 | 199 | 3 | 1.60 | 1.20 | more pairs + system prompt + chat template |
| 3 | 158 | 8 | — | — | judge hit an API quota, no scores |
| 4 | **154** | **8** | **2.60** | **3.20** | dataset cleanup + 8 epochs + judge moved to OpenAI |

### Findings

**1. Growing the dataset helped, but not linearly.** Going from 80 → 199 pairs gained **+0.60 accuracy**. However, the best run (#4) used *fewer* pairs — 154 vs 199 — and still scored higher. Raw pair count is not the lever; coverage and cleanliness are.

**2. Fact coverage matters more than volume.** Run #1's dataset contained **zero** pairs about Yusuf Balasaguni and **zero** about "Кутадгу билиг", even though both appear in the test questions. The model literally could not answer what it had never seen. After fixing generation: Kashgari 8 pairs, Balasaguni 6, Kutadgu 3, Yasawi 5 — and two of five answers became fully correct.

**3. Epoch count fixed exact proper nouns.** At 3 epochs the model produced the garbled "Кутадгы орус турал", even though the correct "Кутадгу билиг" appeared in the dataset three times. At 8 epochs (160 steps instead of 75) it reproduces the title correctly. Rare proper nouns need more passes to overcome the base model's priors.

**4. Cleaning the dataset beat growing it.** Tightening the teacher prompt removed document-structure meta-questions ("which section is this fragment in", "what is the chapter called") — 11 of them, down to 0. The dataset shrank by 45 pairs and accuracy went up.

**5. `question_type` as a `Literal` instead of `str`.** The free-form string field drifted into **44 different spellings** (`кто_такой?`, `что_написал_создал`, `куда_отправлял`…), making question types impossible to balance. The enum restored exactly 4 categories, and "which century" questions rose from 9 to 31.

### Before / after fine-tuning

| Question | Before | After |
|---|---|---|
| Кто такой Махмуд Кашгари? | "an Armenian historian… wrote the 'Chinese Encyclopedia'" | "author of «Дивани лугат ат-тюрк» (Dictionary of Turkic Dialects)" ✅ |
| Какую книгу написал Юсуф Баласагуни? | "wrote 'Манас'" | «Кутадгу билиг» ("Blessed Knowledge") ✅ |
| Кто такой Ходжа Ахмед Яссауи? | "a Kazakh politician of the early 20th century" | "scholar and public figure, author of the 'Book of Wisdom'" ⚠️ |

The base model confidently invented facts (Kashgari as an Armenian historian, Yasawi as a 20th-century politician). After fine-tuning, answers stay within the textbook's subject domain.

---

## Evaluation

Two complementary evaluations, measuring different things:

**1. LLM-as-judge** (notebook section 13) — `gpt-5` scores the *fine-tuned model's* answers 1–5 on accuracy and faithfulness, against the retrieved source chunk. Result: **2.60 / 3.20**.

**2. Golden set** (`data/golden_set.json` + `evals/run_eval.py`) — 10 hand-verified questions where every expected fact was checked against `data/history_text.txt`. Scores two things the judge blurs together:

- **Fact recall** — for answerable questions, did the answer contain the expected facts? Alias lists mean paraphrase isn't penalised.
- **Refusal** — for the 3 questions *not* answerable from pages 3–23 (Balasaguni's century, plus two off-topic probes), did the model correctly decline instead of inventing an answer?

Both directions matter: a model that answers everything scores well on recall and badly on refusal; one that refuses everything does the reverse.

```bash
cd webapp && uvicorn server:app --port 8000   # terminal 1
python evals/run_eval.py                      # terminal 2
```

Latest result against the RAG webapp:

| Metric | Score |
|---|---|
| Fact recall (answerable) | 7/7 (100%) |
| Correct refusals | 3/3 (100%) |
| Project-brief questions | 5/5 (100%) |
| **Overall** | **10/10 (100%)** |

Note this measures the **RAG webapp**, not the fine-tuned model — the harness targets an HTTP endpoint, and the LoRA adapter needs a GPU runtime. To evaluate the fine-tuned model against the same golden set, serve it behind a compatible `/api/ask` endpoint and point `--url` at it.

---

## Known limitations

Stated plainly, since the judge surfaced them:

1. **"В каком веке жил Юсуф Баласагуни?" cannot be answered from pages 3–23.** The extracted text never mentions the 11th century at all (the only centuries present are XIII, XVI, XVIII). The model replies "not stated in the text" — the judge scored that **5/5 faithfulness** against 1/5 accuracy, i.e. correct behavior for a grounded model.
2. **"Кутадгу билиг" is attributed to Kashgari instead of Balasaguni** (1/5). They appear as adjacent list items (4 and 5) in the textbook, and the model conflates them.
3. **Language bleed.** One answer mixed English words and Chinese characters into Russian text — a side effect of sampling-based decoding on a multilingual base model. Fixable with `do_sample=False` in `ask()`.
4. **154 pairs** — above the 100 minimum, below the recommended 200+.

---

## Repository structure

```
├── history_finetuning.ipynb    # full pipeline, with saved cell outputs
├── data/
│   ├── history_text.txt        # OCR output (29,311 characters)
│   ├── history_sft_dataset.json# SFT dataset (154 pairs)
│   ├── golden_set.json         # hand-verified eval set (10 questions)
│   ├── loss_curve.png          # training loss plot
│   └── *.pdf                   # source textbook (gitignored, 51 MB)
├── evals/
│   └── run_eval.py             # scores fact recall + refusal behavior
└── webapp/                     # chat interface (RAG, not the fine-tuned model)
    ├── server.py
    └── static/
```

> The project brief lists `history_text.txt` and `history_sft_dataset.json` as flat deliverables. They live in `data/` here for tidiness — copy them to the root when submitting if your grader expects the flat layout.

## Running it

**Notebook** — Colab, Runtime → T4 GPU. Secrets required: `OPENAI_API_KEY`, `HF_TOKEN`. Then Runtime → Run all.

**Web interface**

```bash
cd webapp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>.

> **Note:** `webapp/` is a separate RAG system — it passes `data/history_text.txt` to GPT-5-mini as context. It does **not** use the fine-tuned model and is not evidence of that model's quality. It is a demo interface.
