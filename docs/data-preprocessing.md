# Data Preprocessing in CaPT

This document describes every data preprocessing step that happens between the raw
SuperGLUE / CommonsenseQA datasets and the tensors fed to the model. It covers both
training pipelines:

- **Encoder–decoder (T5)**: [capt/train_t5.py](../CaPT/capt/train_t5.py) → [src/dataset.py](../CaPT/src/dataset.py) → [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py)
- **Decoder-only (Llama / Qwen)**: [capt/train_llama.py](../CaPT/capt/train_llama.py), [capt/train_qwen.py](../CaPT/capt/train_qwen.py) → [src/superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py)

## Summary

The project does **light, task-formatting style preprocessing**, which is standard for
prompt-tuning research. It does:

1. Download the datasets and cache them as pickles
2. Turn each example into a text prompt using a fixed per-task template (verbalization)
3. Map labels to words (T5) or class indices (Llama/Qwen)
4. Re-split the data: carve a new dev set out of train, and use the official validation set as the test set
5. Tokenize with the model's own tokenizer
6. Truncate to a maximum length
7. Pad each batch dynamically, and build attention masks
8. Shuffle the training data

It does **not** do any text cleaning or normalization (no lowercasing, no stripping
punctuation or stop-words, no deduplication), no data augmentation, no class
rebalancing, and no filtering of long or noisy examples. Raw text goes straight into
the templates. The only normalization step in the code base, `normalize_answer` in
`train_llama.py`, is an evaluation helper (SQuAD-style), not an input preprocessing
step. A `noise_rate` argument exists on `SuperGlueData` but is never used.

---

## 1. Data acquisition

[superglue/get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py)
downloads six SuperGLUE tasks from the HuggingFace `super_glue` dataset and pickles each
split as is:

```
data/superglue/{boolq,cb,copa,multirc,rte,wic}/{train,validation,test}.pkl
```

No transformation happens at this stage. ReCoRD, WSC and AX-* are not downloaded.

**CommonsenseQA** ([src/commonsenseqa_loader.py](../CaPT/src/commonsenseqa_loader.py)) is
loaded directly from HuggingFace (`tau/commonsense_qa`) on every run. If that fails, the
loader falls back to local `data/commonsenseqa/{train,dev}_rand_split.jsonl` files. Only
the `id`, `question`, `choices` and `answerKey` fields are kept.

Note: the official SuperGLUE `test.pkl` files are downloaded but **never used**,
because their labels are hidden (`-1`).

## 2. Input formatting (verbalization)

Each loader function (`boolq()`, `cb()`, …) has two modes, controlled by `text_to_text`.
**Every training script in this repo calls the loaders with `text_to_text=True`**
(`train_t5.py` hard-codes it; `train_llama.py`/`train_qwen.py` pass it into
`get_superglue`), so the text-to-text format below is the one that is actually used.

### Text-to-text templates (used)

| Task | Input template | Label words |
|---|---|---|
| BoolQ | `boolq passage: {passage} question: {question}` | `False` / `True` |
| CB | `cb hyopthesis: {hypothesis}. premise: {premise}` | `entailment` / `contradiction` / `neutral` |
| COPA | `choice1: {choice1} choice2: {choice2} premise: {premise} question: {question}` | `choice1` / `choice2` |
| MultiRC | `question: {question} answer: {answer}. paragraph: {paragraph}` | `False` / `True` |
| RTE | `rte sentence1: {premise} sentence2: {hypothesis}` | `entailment` / `not_entailment` |
| WiC | `wic sentence1: {sentence1} sentence2: {sentence2} word: {word}` | `False` / `True` |
| CommonsenseQA | `commonsenseqa question: {question} choices: A: …, B: …, C: …, D: …, E: …` | `A`–`E` |

These follow the T5 convention of starting each input with the task name. A few
quirks are reproduced exactly as they appear in the code:

- COPA and MultiRC define a `text_format` with a task prefix (`copa …`, `multirc …`)
  but then build the string with an f-string that **omits** it, so those two tasks have
  no task prefix.
- CB's template spells "hypothesis" as `hyopthesis`. Because prompt-tuning learns from
  the text it is given, this has little effect, but it is not the standard T5 template.
- For COPA, `question` is the raw SuperGLUE field value (`cause` or `effect`).

### Classification-mode formatting (defined, but not used)

With `text_to_text=False`, loaders return sentence pairs, e.g. `(premise, hypothesis)`.
For COPA they return two (context, choice) pairs per example with a `because`/`so` joiner
(written `beacuse` in the code). WiC becomes
`"{s1} {s2} Does {word} have the same meaning in both sentences?"`. `SuperGlueData`
supports this mode with `build_inputs_with_special_tokens` / token-type (segment) IDs
and pads the two COPA candidates to equal length. None of the current scripts call this
path; it is inherited from the SMoP code base.

### Decoder-only templates (defined, but effectively not used)

`train_llama.py`/`train_qwen.py` define `TASK_CONFIGS` with natural-language templates
such as:

```
Premise: {premise}
Hypothesis: {hypothesis}
Question: Does the premise entail the hypothesis?
Answer:
```

However, `LlamaClassificationDataset.format_input()` only applies these templates when
the input is a **tuple**. Because the data is loaded with `text_to_text=True`, each
input is already a **string**, so `format_input` falls through to `return str(input_seq)`.
**In practice, Llama and Qwen receive the same T5-style strings shown in the table
above**, not the `Premise: … Answer:` templates. The tuple-parsing branches (which split
WiC sentences at the midpoint of the word list, find COPA questions by searching for the
substrings "because"/"so", and so on) are never reached.

## 3. Label processing

- **T5**: the label word is tokenized with the EOS token appended
  (`tokenize(f"{label}{eos_token}")`) and used as the decoder target sequence. In the
  collate function, label sequences are padded with `-100` so padded positions are
  ignored by the cross-entropy loss. At evaluation time, `-100` is replaced with the pad
  ID and predictions and gold labels are decoded back to strings and compared as exact
  strings.
- **Llama / Qwen**: the label word is mapped to an integer class index through
  `TASK_CONFIGS[task]['label_map']` (e.g. `entailment → 0`, `contradiction → 1`,
  `neutral → 2` for CB). The model is trained as a sequence classifier with
  `num_labels` outputs. An unknown label raises a `ValueError`.

## 4. Train / dev / test re-splitting

Because SuperGLUE test labels are not public, the code builds its own three-way split
in `get_superglue()`:

| Mode | Train | Dev | Test |
|---|---|---|---|
| text-to-text (used by all scripts) | first 90% of the official train set, **per label** | last 10% of the official train set, **per label** | full official validation set |
| classification (unused) | full official train set | first half of the official validation set | second half of the official validation set |

Details of the text-to-text split (`construct_dev_from_train_t2t`):

- Examples are grouped by label string, and each group is cut at `n * 9 // 10`. The
  result is a **stratified** 90/10 split that keeps the class ratios.
- The cut is **deterministic** and follows HuggingFace's file order. It is not random,
  so it does not depend on the seed.
- After the split, the training list is ordered by label (all examples of label A,
  then all of label B). The shuffling described in step 7 removes this ordering.
- The `num_labels == 1` branch (an 8/9 non-stratified split, intended for regression)
  is never reached for these tasks.
- CommonsenseQA uses its own **non-stratified** 90/10 split
  (`construct_dev_from_train_t2t_csqa`), and the official validation set is its test set.

Caveat: for MultiRC (and BoolQ to a lesser extent), several examples share the same
paragraph. The split is per example, not per paragraph, so the same paragraph can
appear in both train and dev. This only affects model selection on dev, not the
reported test score.

## 5. Tokenization and truncation

### T5 (`SuperGlueData.formatting`)

- Tokenizer: `AutoTokenizer.from_pretrained(..., model_max_length=max_length)`. If it
  has no pad token, `[PAD]` is added and the embedding matrix is resized.
- Text → `tokenize` → `convert_tokens_to_ids` → `build_inputs_with_special_tokens`,
  which appends `</s>` for T5.
- **Length budget**: `max_length = args.max_length − num_virtual_tokens` (512 − 1 = 511
  with the defaults), so that the input plus the capsule prompt fits in 512 positions.
  For `prefix-tuning`/`full`, the budget is the full `max_length`.
- **Truncation**: the ID list is cut to `input_ids[:max_length]` *after* special tokens
  are added. For long inputs (mainly BoolQ and MultiRC passages), the trailing `</s>` is
  removed and the end of the input is lost. Because the template puts the question
  first in MultiRC and last in BoolQ, a long BoolQ passage can push the
  **question itself** out of the window. `segment_ids` is not truncated along with the
  IDs; it is not passed to the T5 model, so this is harmless.
- Label sequences are not truncated (they are only a few tokens long).

### Llama / Qwen (`LlamaClassificationDataset.create_dataset`)

- If the tokenizer has no pad token, it reuses the EOS token
  (`pad_token = eos_token`).
- Standard `tokenizer(text, max_length=max_length, truncation=True, padding=False)`,
  which adds the BOS token for Llama. This is right-side truncation at 512 tokens, with
  the same risk of losing the end of a long BoolQ/MultiRC input.
- No length budget is reserved for the virtual token(s).

## 6. Batching, padding and attention masks

Padding is **dynamic**: each batch is padded only to its longest sequence.

- **T5** (`collate_fn` in `prepare_dataset`): `pad_sequence(..., padding_value=pad_token_id)`
  for inputs and `-100` for labels. The attention mask is computed in the training loop
  as `input_ids != pad_token_id`.
- **Llama / Qwen** (`pad_sequence_collate`): **right-padding** with `pad_token_id`, and
  the attention mask is extended with zeros.

## 7. Shuffling and seeding

- **T5**: `SuperGlueData.create_dataset()` calls `random.shuffle(self.dataset)` on every
  split. `train_t5.py` never seeds Python's `random` module, so this first shuffle is
  **not reproducible** across runs. The training `DataLoader` then shuffles again with a
  `torch.Generator` seeded from `--random_seed`, but because its starting order differs
  from run to run, batch composition still varies even with the same seed. Shuffling the
  dev/test sets has no effect on the metrics. On Windows, `num_workers=0`.
- **Llama / Qwen**: `seed_everything()` seeds `random` and `numpy`, the dataset is not
  pre-shuffled, and only the training `DataLoader` shuffles (with the seeded generator),
  so the order is reproducible. `num_workers=4` is hard-coded, which can be slow or fail
  on Windows.

## 8. Evaluation-side processing

| Task | Metric (`score()`) |
|---|---|
| CB | mean of accuracy and macro-F1 |
| MultiRC | `f1a`, computed as **macro-F1 over all answer options** (not the official F1a, which is binary F1 on the positive class) and without EM |
| all others | accuracy |

For T5, the generated text is decoded with `skip_special_tokens=True` and compared
with the gold label as an exact string. A generation that is not one of the label words
is simply counted as wrong.

---

## Quick reference: what is and isn't done

| Technique | Used? | Where |
|---|---|---|
| Dataset download/caching | Yes | `get_huggingface_superglue.py` |
| Template / prompt verbalization | Yes | `superglue_loader_*.py`, `commonsenseqa_loader.py` |
| Label verbalization (words) / label → index | Yes | loaders / `TASK_CONFIGS` |
| Stratified train→dev split | Yes (90/10) | `construct_dev_from_train_t2t` |
| Official validation used as test | Yes | `get_superglue` |
| Subword tokenization | Yes | HF `AutoTokenizer` |
| Truncation | Yes (right-side, max 512 or 511) | `dataset.py`, `train_llama.py` |
| Dynamic padding + attention mask | Yes | collate functions |
| Loss masking of padded labels (`-100`) | Yes (T5) | `collate_fn` |
| Shuffling | Yes | `create_dataset` / `DataLoader` |
| Lowercasing / punctuation / stop-word removal | **No** | — |
| Text cleaning, deduplication, outlier filtering | **No** | — |
| Data augmentation / noise injection | **No** (`noise_rate` unused) | — |
| Class balancing / re-weighting | **No** | — |
| Sliding window / chunking for long passages | **No** | — |
