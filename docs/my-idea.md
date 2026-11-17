# My Project Idea: Seeing Capsules for Bangla

**CaPT for English→Bangla multimodal translation**

> **Related docs:** [gaps.md](gaps.md) lists all the research gaps, [document.md](document.md) explains the code, and [dataset.md](dataset.md) explains the data.
> **Time:** 1 month · **Compute:** free Colab and Kaggle (T4 / P100, 16 GB)

> **Note on sources:** The dataset sizes and download details below are from memory and have **not** been checked yet. Confirm them on the official dataset page before starting (see [Section 9](#9-to-check-before-starting)).

---

## Table of Contents

1. [The Idea in One Paragraph](#1-the-idea-in-one-paragraph)
2. [Which Gaps It Covers](#2-which-gaps-it-covers)
3. [Research Questions](#3-research-questions)
4. [Data: Bengali Visual Genome](#4-data-bengali-visual-genome)
5. [Model](#5-model)
6. [Method](#6-method)
7. [Experiments](#7-experiments)
8. [Four-Week Plan](#8-four-week-plan)
9. [To Check Before Starting](#9-to-check-before-starting)
10. [Risks and Backup Plans](#10-risks-and-backup-plans)
11. [Where to Edit the Code](#11-where-to-edit-the-code)

---

## 1. The Idea in One Paragraph

CaPT was only tested on **English classification** with **text-only** input. This project tests it on a harder setting: **translating English image captions into Bangla, with the image as extra input**. The model must **write a sentence** (generation), in a **low-resource language** (Bangla), using **two kinds of input** (text and image). The capsule already averages the input tokens. If one of those tokens is the image, the capsule becomes "image-aware" at almost no extra cost. A capsule is also added to the **decoder**, the part of the model that writes the Bangla output.

[Back to top](#table-of-contents)

---

## 2. Which Gaps It Covers

| Gap | From [gaps.md](gaps.md) | How this project covers it |
|---|---|---|
| **6** | [Only classification tasks are tested](gaps.md#gap-6-only-classification-tasks-are-tested) | The output is a full Bangla sentence, scored with chrF++ and BLEU. |
| **9** | [Only English, in-domain data](gaps.md#gap-9-only-english-in-domain-data) | The target language is Bangla, and a separate **challenge test set** is used as a harder, shifted test. |
| **13** | [T5's decoder gets no capsule](gaps.md#gap-13-t5s-decoder-gets-no-capsule) | Capsules are added to the decoder, which writes the Bangla text. |
| **New** | Multimodal input | The image becomes part of what the capsule averages. |
| **11** (bonus) | [Mean pooling includes padding](gaps.md#gap-11-mean-pooling-is-very-simple-and-it-includes-padding) | The capsule averages only real tokens (masked mean). |

[Back to top](#table-of-contents)

---

## 3. Research Questions

**Main question:**
> *Can a single image-aware capsule improve parameter-efficient English→Bangla translation, especially when the English word is ambiguous?*

**Sub-questions:**

1. Does CaPT still help when the output is a sentence, not a label? (Gap 6)
2. Does a capsule in the decoder help, compared with the encoder only? (Gap 13)
3. Does adding the image to the capsule improve translation?
4. Does the model really **use** the image, or would any image work just as well?
5. Do the gains hold on the harder challenge test set? (Gap 9)

[Back to top](#table-of-contents)

---

## 4. Data: Bengali Visual Genome

**Bengali Visual Genome (BVG)** was released at LREC 2022 and used in the WAT multimodal translation shared tasks. Each example has:

- an English caption,
- its Bangla translation,
- an image, with a **box** marking the region the caption describes.

| Split | Size (approx.) | Use |
|---|---|---|
| Train | ~29,000 | Training |
| Dev | ~1,000 | Choosing the best checkpoint and learning rate |
| Evaluation test | ~1,600 | Main results |
| **Challenge test** | ~1,400 | Sentences with **ambiguous English words** where the image shows the right meaning. This tests image use and serves as the harder test (Gap 9). |

BVG also contains **Bangla image captioning** data (image → Bangla caption). This is an optional second task (Experiment 8).

**Where to get it:** the LINDAT/CLARIN repository. Check the license before using it.

**Image features:** Run a frozen CLIP model (for example `openai/clip-vit-base-patch32`) **once** on the **cropped region** of every image, and save the features to disk (`.npy` or `.pt`). This takes about 20 minutes on a T4. Training then reads only the saved vectors, so no vision model runs during training.

[Back to top](#table-of-contents)

---

## 5. Model

Plain T5 cannot read or write Bangla script, so the model must change.

| Option | Pros | Cons |
|---|---|---|
| **mT5-Base** (`google/mt5-base`) **(main choice)** | Handles both English and Bangla. Its weights load into `T5ForConditionalGeneration`, so the existing CaPT code in `modeling_t5.py` works. Fits on a T4. | Prompt-tuning methods are known to be unstable on small mT5 models, so the learning rate needs tuning. |
| **BanglaT5** (`csebuetnlp/banglat5`) **(backup)** | Strong at generating Bangla. Uses the same T5 architecture. | Its tokenizer is built for Bangla. Check how well it handles the **English source** before using it. |

`csebuetnlp/banglat5_nmt_en_bn` is already fine-tuned for English→Bangla. Use it only as a **reference score**, not as the frozen base model.

[Back to top](#table-of-contents)

---

## 6. Method

The method stays as small as CaPT itself. All of the base model stays **frozen**.

```
Input:   [image token] [English caption tokens ...]
                 │
Encoder: every layer adds a capsule  c_enc = p_enc + masked_mean(tokens)
                 │                      (tokens include the image token)
Decoder: every layer adds a capsule  c_dec = p_dec + masked_mean(encoder output)
                 │
Output:  Bangla sentence
```

**Parts:**

1. **Image token.** The saved CLIP vector goes through **one shared linear layer** (CLIP size → model hidden size, about 400K parameters) and is placed before the text as one extra token.
2. **Encoder capsule.** This is the original CaPT: `p + mean(tokens)`. Because the image token is part of the input, the capsule now contains image information automatically.
3. **Decoder capsule (Gap 13).** One learned vector per decoder layer, plus the mean of the encoder output. It carries both the source text and the image to the part of the model that writes Bangla.
4. **Masked mean (Gap 11).** Average only the real tokens, not padding.

**Trainable parameters:** one vector per encoder layer, one per decoder layer, and the linear layer for the image. Report this count next to every result, as the paper does.

[Back to top](#table-of-contents)

---

## 7. Experiments

Run each setting with **3 seeds** and report **mean ± standard deviation**.

| # | Setting | What it answers |
|---|---|---|
| 1 | Frozen mT5, no training | Lower bound |
| 2 | Prompt-Tuning (length 20) | Baseline from the paper |
| 3 | CaPT, text only, encoder only | Does CaPT work for generation? (Q1, Gap 6) |
| 4 | CaPT, text only, encoder + decoder | Does the decoder capsule help? (Q2, Gap 13) |
| 5 | CaPT + image token, encoder + decoder | Does the image help? (Q3) |
| 6 | Experiment 5's model, but with **mismatched images** at test time | Does the model really use the image? (Q4) |
| 7 | Experiments 3–6 on the **challenge test set** | Do the gains hold where the image matters most? (Q5, Gap 9) |
| 8 *(optional)* | Bangla image captioning (image → Bangla) | A second generation task |
| — | `banglat5_nmt_en_bn` | Reference score from a fully fine-tuned model |

**Notes:**
- **Experiment 6** needs no training, only evaluation, so it is cheap. If the score barely drops with the wrong image, report that honestly. It is still a finding.
- **Optional extra comparison:** CLIP features from the **cropped region** against the **whole image**.

**Metrics:**
- **chrF++** (main metric). It suits Bangla better than BLEU, because Bangla words change form a lot.
- **BLEU** with `sacrebleu`, for comparison with other papers.
- *(Optional)* **BERTScore** with a multilingual model.
- **Trainable parameters** and **training time** for every setting.

[Back to top](#table-of-contents)

---

## 8. Four-Week Plan

| Week | Tasks | Output |
|---|---|---|
| **1** | Set up on Kaggle or Colab (`pip install -e .` on the bundled transformers). Load mT5 into the CaPT T5 code and check that generation works. Write the BVG loader. Save the CLIP region features. Add chrF++ and BLEU. Do a quick learning-rate sweep. Run Experiments 1–2. | Working pipeline and baselines |
| **2** | Add the masked mean. Run Experiment 3, then add decoder capsules and run Experiment 4. | Text-only results |
| **3** | Add the image token and linear layer. Run Experiments 5–6. | Multimodal results |
| **4** | Run Experiment 7 (and 8 if time allows). Make tables and write the report. | Final report |

**Minimum result:** If time runs short, Experiments 1–4 alone (CaPT for Bangla generation with a decoder capsule) are a complete small project. The image part is an extra layer on top.

**Free GPU tips:**
- Use **Kaggle** as the main machine: about 30 GPU hours per week, and sessions can run for up to 12 hours.
- Use **Colab** for quick debugging.
- **Save a checkpoint every epoch.** Free sessions can stop without warning.
- Save the CLIP features and tokenized data **once**, and upload them as a Kaggle dataset so every session can reuse them.

[Back to top](#table-of-contents)

---

## 9. To Check Before Starting

- [ ] BVG: exact split sizes, download link and license.
- [ ] BVG: whether the challenge test set includes the image and box, like the other splits.
- [ ] mT5-Base loads into the CaPT `T5ForConditionalGeneration` and produces sensible Bangla output.
- [ ] BanglaT5 tokenizer: how many English words become unknown tokens (only needed if mT5 does not work).
- [ ] Related work: search for "Bengali Visual Genome", "WAT multimodal translation" and papers that cite CaPT. Note their chrF++ and BLEU scores so your baselines can be compared with them.
- [ ] Reproduce original CaPT on T5-Base + COPA first (see [gaps.md, section 7](gaps.md#7-practical-tips-for-starting)).

[Back to top](#table-of-contents)

---

## 10. Risks and Backup Plans

| Risk | Backup plan |
|---|---|
| mT5 with prompt-style training gives very low scores | Try a wider learning-rate range, or switch to BanglaT5. The comparison **between** methods is still valid even if all the scores are low. |
| BVG cannot be downloaded, or the license is unclear | Use BAN-Cap (Bangla captions for Flickr8k) and do **captioning** instead of translation. |
| The image gives no improvement | Report it, and use Experiment 6 and the challenge test set to explain why. A clear negative result is still useful. |
| Free GPU time runs out | Drop to 2 seeds, or use mT5-Small for the early experiments. |

[Back to top](#table-of-contents)

---

## 11. Where to Edit the Code

| What to change | Where |
|---|---|
| Capsule pooling (masked mean) | `CaPT` class in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py#L664) |
| Add capsules to the decoder | `T5Block.forward` in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py#L752) (currently skipped when `self.is_decoder`) |
| Image token and linear layer | Encoder input embedding step in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py) |
| BVG data loader | New file next to [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py) (see [dataset.md, section 10](dataset.md#10-adding-a-new-dataset)) |
| Metrics (chrF++, BLEU) | `score()` in [capt/train_t5.py](../CaPT/capt/train_t5.py#L403) |

**Remember:** After editing the `transformers/` folder, reinstall it, or use `pip install -e .` once so that your edits apply immediately.

[Back to top](#table-of-contents)
