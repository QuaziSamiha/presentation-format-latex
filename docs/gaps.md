# CaPT Research Gaps and Project Ideas

What CaPT solves, what it still leaves open, and how you can turn each open gap into a new project.

> **Related docs:** [document.md](document.md) explains the code, and [dataset.md](dataset.md) explains the data.
> **Easy paper summary:** [capt-summary/](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md)
> **Code folder:** [CaPT/](../CaPT/)

> **Note on sources:** This guide is based on the paper summary, the two docs above and the released code. The paper's appendices (§S3, §S4) were not checked, so a few gaps may already be partly covered there. Before you start a project, read those appendices and search for newer papers that cite CaPT.

---

## Table of Contents

1. [What Is a Research Gap?](#1-what-is-a-research-gap)
2. [Quick Summary Table](#2-quick-summary-table)
3. [Part A: Gaps That CaPT Fills](#3-part-a-gaps-that-capt-fills)
4. [Part B: Gaps That CaPT Leaves Open](#4-part-b-gaps-that-capt-leaves-open)
   - [B1. Scope: where CaPT has not been tested](#b1-scope-where-capt-has-not-been-tested)
   - [B2. Method design: how the capsule is built](#b2-method-design-how-the-capsule-is-built)
   - [B3. Evidence and evaluation: how strong the proof is](#b3-evidence-and-evaluation-how-strong-the-proof-is)
5. [How to Choose Your Project](#5-how-to-choose-your-project)
6. [Five Ready-Made Project Ideas](#6-five-ready-made-project-ideas)
7. [Practical Tips for Starting](#7-practical-tips-for-starting)

---

## 1. What Is a Research Gap?

A **research gap** is a question that existing work has not answered yet, or has answered only partly.

You can usually find gaps in four places:

| Where to look | Question to ask | Example in CaPT |
|---|---|---|
| **What was not tested** | "Does it still work on X?" | CaPT was never tested on text generation. |
| **Simple design choices** | "Is there a better way to do this step?" | The capsule uses a plain average of the input. |
| **Weak evidence** | "Is the claim really proven?" | The "attention anchor" is shown with pictures, not measurements. |
| **Unexplained results** | "Why does this happen?" | More capsules make results worse, but nobody explains why. |

A good project takes **one** gap, asks a clear question, and answers it with experiments.

[Back to top](#table-of-contents)

---

## 2. Quick Summary Table

**Difficulty:** 🟢 easy (a few weeks, small GPU) · 🟡 medium (1–3 months, one good GPU) · 🔴 hard (several months, or big GPUs).

| # | Gap | Type | Difficulty |
|---|---|---|---|
| 6 | Only classification tasks are tested | Scope | 🟡 |
| 7 | Only small models (≤ 1.5B) are tested | Scope | 🔴 |
| 8 | Gains are weaker on decoder-only models | Scope | 🟡 |
| 9 | Only English, in-domain data | Scope | 🟢 |
| 10 | No few-shot, multi-task or transfer tests | Scope | 🟢 |
| 11 | Mean pooling is very simple and includes padding | Design | 🟢 |
| 12 | The "two-way anchor" does not fully work in causal models | Design | 🟡 |
| 13 | T5's decoder gets no capsule | Design | 🟢 |
| 14 | No explanation for why longer capsules hurt | Design | 🟡 |
| 15 | Variant study is incomplete and not released | Design | 🟢 |
| 16 | Not combined with LoRA or adapters | Design | 🟡 |
| 17 | Attention-anchor proof is only visual | Evidence | 🟡 |
| 18 | Finding II tested on one model, two tasks | Evidence | 🟢 |
| 19 | Statistics are fragile (tiny datasets, 3 runs) | Evidence | 🟢 |
| 20 | Learning rate still needs searching | Evidence | 🟢 |
| 21 | Some comparisons are not fully fair | Evidence | 🟡 |
| 22 | LoPA is still more accurate | Evidence | 🟡 |
| 23 | Inference cost is not measured | Evidence | 🟢 |
| 24 | Nobody knows what the capsule learns | Evidence | 🟡 |
| 25 | Input wording for decoder models is untested | Evidence | 🟢 |

Numbers 1–5 are the gaps in **earlier** work that CaPT fills (Part A).

[Back to top](#table-of-contents)

---

## 3. Part A: Gaps That CaPT Fills

These are the problems in **earlier** methods that the paper says it solves. You need them for the "Related Work" or "Background" part of your own project, and they explain why CaPT matters.

### Gap 1. Soft prompts ignore the input
Normal soft prompts only learn *"what job to do"* (task-aware). They are the same for every sentence. In attention heatmaps, they mostly look at each other and barely look at the input words.
**CaPT's answer:** Add the average of the input to the prompt, so the prompt also knows *"what this sentence is about"* (instance-aware).

### Gap 2. Searching for prompt length is expensive
With Prompt-Tuning and P-Tuning v2, you must try many lengths (5, 10, 20, 100 …) and train a model for each. This makes total training time about **8–9× higher** than CaPT.
**CaPT's answer:** Always use length 1. Nothing to search.

### Gap 3. Too many parameters
Deep prompt-tuning trains about **1 million** parameters on T5-Base.
**CaPT's answer:** One vector per layer, about **9,216** parameters (≈ 0.004%).

### Gap 4. The role of attention in prompt learning is under-studied
Many papers study attention in Transformers, but few look at how **prompts** and **input words** pay attention to each other.
**CaPT's answer:** Studies this with heatmaps and introduces the idea of an **"attention anchor"**.

### Gap 5. Instance-aware prompt methods are costly
Methods like M-IDPG and LoPA also use input information, but they add extra networks and extra settings to search. Their total training time is about **12–15×** that of CaPT.
**CaPT's answer:** Get instance information with a simple average, with no extra network.

[Back to top](#table-of-contents)

---

## 4. Part B: Gaps That CaPT Leaves Open

Each gap below has the same five parts:

- **What it means:** the gap in plain words.
- **Why it matters:** why someone would care.
- **Evidence:** where you can see the gap in the paper or code.
- **How you can contribute:** concrete project ideas.
- **What you need:** difficulty and compute.

---

### B1. Scope: where CaPT has not been tested

#### Gap 6. Only classification tasks are tested

**What it means:** Every dataset (BoolQ, CB, COPA, MultiRC, RTE, WiC, CommonsenseQA) asks the model to pick one answer from a small fixed list. Nobody knows whether CaPT works when the model must **write** a long answer.

**Why it matters:** Most real LLM use is generation: summaries, translations, chat answers, code. If CaPT only helps classification, it is much less useful.

**Evidence:** [dataset.md, section 1](dataset.md#1-overview) says "Every task is a classification problem".

**How you can contribute:**
- Test CaPT on **summarisation** (for example XSum or CNN/DailyMail) using T5. T5 already generates text, so the code needs only small changes.
- Test CaPT on **generative question answering** (for example SQuAD, where the answer is written, not picked).
- Test CaPT on **reasoning tasks** such as GSM8K (math word problems).
- Compare against Prompt-Tuning and LoRA on the same tasks, and report which works best.

**A possible research question:** *"Does a single capsule prompt still help when the output is long text instead of a single label?"*

**What you need:** 🟡 Medium. T5-Base on one GPU (16–24 GB) is enough to start. The new work is mainly data loaders and metrics (ROUGE, BLEU, exact match).

---

#### Gap 7. Only small models are tested

**What it means:** The biggest model in the paper is Qwen-2.5-1.5B. Today's popular models have 7B, 13B, 70B or more parameters.

**Why it matters:** One vector of 2,048 numbers might be enough to guide a 1B model, but perhaps not a 70B model. It might also work **better** on big models, as normal prompt-tuning does. Nobody knows.

**Evidence:** [document.md, section 7](document.md#7-technology-used) lists the four models used. The largest is 1.5B.

**How you can contribute:**
- Run CaPT on a **7B or 8B** model (for example Llama-3.1-8B or Qwen-2.5-7B) and draw a **scaling curve**: model size on the x-axis, score on the y-axis.
- To save memory, load the frozen model in 4-bit or 8-bit (as in QLoRA) and train only the capsules.
- Test a **mixture-of-experts** model, which routes tokens differently.

**A possible research question:** *"How does CaPT's advantage over prompt-tuning change as model size grows?"*

**What you need:** 🔴 Hard for compute. You need at least one 40–80 GB GPU, or a cloud GPU. With 4-bit loading, an 8B model can fit on a 24 GB GPU.

---

#### Gap 8. Gains are weaker on decoder-only models

**What it means:** CaPT does very well on T5 (it even beats full fine-tuning on T5-Large). On Llama and Qwen the improvement is smaller. The paper says this but does not explain it.

**Why it matters:** Almost all modern LLMs are decoder-only. If CaPT is weaker there, that is a big limitation.

**Evidence:** [Paper summary, section 4.2](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md) says the gains on Llama and Qwen "are smaller than on T5 models". See also Gap 12, which may be the reason.

**How you can contribute:**
- **Find the cause.** Try each of these and see which one closes the gap:
  - Remove `.detach()` so gradients can flow through the average.
  - Move the capsule to a different position (see Gap 12).
  - Use a better pooling method (see Gap 11).
- Compare against full fine-tuning on Llama, not just "Linear Head" (see Gap 21).

**A possible research question:** *"Why is CaPT less effective in decoder-only models, and how can we fix it?"*

**What you need:** 🟡 Medium. Llama-3.2-1B fits on a 24 GB GPU.

---

#### Gap 9. Only English, in-domain data

**What it means:** All data is English, and the model is always tested on the same kind of data it was trained on.

**Why it matters:** Real systems meet new languages and new kinds of text. A method that only works "in-domain" is less trustworthy.

**Evidence:** All datasets in [dataset.md, section 2](dataset.md#2-supported-datasets) are English.

**How you can contribute:**
- **Multilingual:** Train CaPT on mT5 or Qwen with a multilingual dataset such as XNLI. Check whether a capsule trained on English works on other languages (zero-shot cross-lingual transfer).
- **Your own language:** If you speak a low-resource language (for example Bangla), build or use a small classification dataset in it. This can be a strong and original contribution.
- **Out-of-domain:** Train on RTE and test on a different entailment dataset, such as SNLI or MNLI.

**What you need:** 🟢 Easy to medium. Small models are enough.

---

#### Gap 10. No few-shot, multi-task or transfer tests

**What it means:** Every model is trained on one full dataset at a time. The paper does not check:
- **Few-shot:** What if you have only 16, 32 or 100 training examples?
- **Multi-task:** Can one set of capsules learn several tasks at once?
- **Transfer:** Can a capsule trained on task A be a good starting point for task B?

**Why it matters:** Prompt methods are often used when data is scarce, and there are many tasks. Older work (for example SPoT) showed that prompt transfer helps a lot.

**How you can contribute:**
- Make **learning curves**: train with 16, 64, 256 and 1,000 examples, and compare CaPT with prompt-tuning and LoRA.
- Train on MNLI first, then use those capsules to start RTE or CB training. Measure the gain.
- Train **one** set of capsules on all six SuperGLUE tasks together.

**What you need:** 🟢 Easy. The code already supports the datasets. You mainly change how many examples you load (see `create_dataset(debug=True)` in [src/dataset.py](../CaPT/src/dataset.py) for a starting point).

---

### B2. Method design: how the capsule is built

#### Gap 11. Mean pooling is very simple, and it includes padding

**What it means:** The instance-aware part is a plain average of all token vectors. Every word counts the same: "the", "a" and padding count as much as "not" or "amazing".

There is also a small detail in the code. The average is taken over **every position**, including padding tokens. When a short sentence is in a batch with long ones, its average is pulled toward the padding vectors.

**Why it matters:** A smarter summary of the input could give a better capsule at almost no extra cost.

**Evidence:** In [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py#L664-L696) and [modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py#L847-L878), the code is `torch.mean(sequence, dim=1, keepdim=True)`. No attention mask is passed in.

**How you can contribute:**
- **Masked mean:** Average only the real tokens, not padding. This is an easy first experiment, and it may give a free improvement.
- **Max pooling**, or **mean + max** together.
- **Attention pooling:** Learn one small "query" vector that decides how much weight each token gets. This adds only one more vector per layer.
- **Weighted by importance:** Give more weight to tokens that already get high attention.
- Report the accuracy **and** the extra parameters for each option, as the paper does in Table 3.

**A possible research question:** *"Can a smarter way of summarising the input improve CaPT without losing its parameter efficiency?"*

**What you need:** 🟢 Easy. This is a good **first project**: the change is a few lines in the `CaPT` class, and T5-Base on COPA/RTE/CB trains quickly.

---

#### Gap 12. The "two-way anchor" does not fully work in causal (decoder-only) models

**What it means:** The paper's key idea is that the capsule and the input words **look at each other** (two-way attention).

- In **T5's encoder**, every token can see every other token, so this works.
- In **Llama and Qwen**, attention is **causal**: a token can only look at tokens **before** it. The capsule sits at **position 0**, the very first position, so it can only see **itself**. The input words can look at the capsule, but the capsule cannot look at them.

The capsule still carries input information, because it is built from the average of all tokens. But the "capsule looks at important input words" half of the anchor cannot happen.

The Llama code also uses `hidden_states.detach()` before the average. This means training cannot learn **through** the average. Only the learned vector `p` and the classifier head learn.

**Why it matters:** This may explain Gap 8 (weaker results on Llama and Qwen). Fixing it could make CaPT much stronger on modern LLMs.

**Evidence:** [modeling_llama.py, LlamaModel.forward](../CaPT/transformers/models/llama/modeling_llama.py#L950-L1001) puts the capsule at `hidden_states[:, :self.prompt_len, :]` and computes it from `hidden_states.detach()`. This point comes from reading the code. The paper does not discuss it.

**How you can contribute:**
- **Move the capsule to the end** of the input (just before the classification token). There it can see every input token.
- **Let the capsule see everything:** Change the attention mask so position 0 can attend to all tokens, while the other tokens stay causal.
- **Remove `.detach()`** and check whether gradients through the average help or hurt.
- Measure attention in both directions for each version (see Gap 17).

**A possible research question:** *"Where should the capsule be placed in a causal language model so that it can act as a true two-way attention anchor?"*

**What you need:** 🟡 Medium. You must change the attention mask code carefully, but Llama-3.2-1B on CB/RTE is small enough for one GPU.

---

#### Gap 13. T5's decoder gets no capsule

**What it means:** In T5, CaPT is only added to the **encoder** (the part that reads). The **decoder** (the part that writes the answer) is unchanged.

**Why it matters:** For classification, the decoder writes only one or two words, so this may not matter. For **generation** tasks (Gap 6), the decoder does most of the work, and a capsule there could help.

**Evidence:** [document.md, section 5.2](document.md#52-how-capt-is-plugged-into-t5) says capsules are added only when `not self.is_decoder`.

**How you can contribute:**
- Add capsules to the decoder layers too. Try building the decoder capsule from the **encoder's output** (cross-attention information) or from the decoder's own tokens.
- Compare: encoder only, decoder only, and both.
- Do this on a generation task, where the decoder matters more.

**What you need:** 🟢 Easy to medium. The T5 code is already set up for this. Combine it with Gap 6.

---

#### Gap 14. No explanation for why longer capsules hurt

**What it means:** One capsule is best. With 20 capsules, Llama drops from 67.51 to 59.38. The paper shows this but does not explain **why**.

**Why it matters:** This goes against the usual belief that "more prompts = more capacity = better". Understanding it would be a real scientific contribution.

**Evidence:** [Paper summary, "CaPT Length"](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md). In the code, when `prompt_len > 1`, the capsules come from `AdaptiveAvgPool1d`, which averages **chunks** of the sentence instead of the whole sentence.

**How you can contribute:** Test a few possible explanations, one at a time:
- **Hypothesis 1, attention dilution:** Several capsules split the input's attention between them, so each one is a weaker anchor. *Test:* measure how much attention each capsule receives.
- **Hypothesis 2, the pooling method:** Chunk averages (from `AdaptiveAvgPool1d`) are noisier than a whole-sentence average. *Test:* use several capsules that all start from the **whole-sentence** mean, each with its own learned vector.
- **Hypothesis 3, optimisation:** More capsules are harder to train with the same learning rate. *Test:* tune the learning rate separately for each length.

**What you need:** 🟡 Medium. The experiments are cheap, but you need careful analysis and clear writing.

---

#### Gap 15. The variant study is incomplete, and the variants are not released

**What it means:** The paper tests four ways to combine the task part and the input part: Addition, Prepending, Extraction and Projection. Only **Addition** is in the released code. Projection actually scores a bit higher (77.64 vs 77.51), but it uses 17.5× more parameters.

**Why it matters:**
- Other people cannot reproduce the variant results.
- There may be a combination method that is as accurate as Projection and as small as Addition.

**Evidence:** [document.md, section 10](document.md#10-command-line-options-explained) says the Prepending, Extraction and Projection variants are **not** in the code folder.

**How you can contribute:**
- **Re-implement** the three variants and publish them. Reproduction studies are valued, and this is a great way to learn the code.
- Try **new cheap combinations**:
  - A learned **gate**: `capsule = g · p + (1 − g) · mean(input)`, where `g` is a single number per layer.
  - **Scaling**: `capsule = p + α · mean(input)`, with a learned `α`.
  - **Element-wise product**: `capsule = p ⊙ mean(input)`.

**What you need:** 🟢 Easy. Small code changes in the `CaPT` class, and T5-Base is enough.

---

#### Gap 16. CaPT is not combined with LoRA or adapters

**What it means:** CaPT is compared with other PEFT methods, but nobody tries **using them together**. CaPT changes the input of each layer, while LoRA changes the weights. These are different, so they might add up.

**Why it matters:** CaPT adds only about 9K parameters. If it raises LoRA's score at almost no cost, that is a useful practical result.

**How you can contribute:**
- Train **LoRA alone**, **CaPT alone** and **LoRA + CaPT**, on the same tasks.
- Use the Hugging Face `peft` library for LoRA on top of the CaPT model. Check that it works with the modified `transformers` version (4.43.2).
- Report the score, the trainable parameters and the training time for each.

**What you need:** 🟡 Medium. Mainly the effort of making two libraries work together.

---

### B3. Evidence and evaluation: how strong the proof is

#### Gap 17. The attention-anchor proof is only visual

**What it means:** The "attention anchor" claim is supported by **heatmap pictures** of a few examples. There is no number that measures it, and no test showing that the anchor **causes** the better accuracy.

**Why it matters:** "We see red squares in the picture" is weak evidence. A picture can be cherry-picked, and something that happens alongside a better score is not proof that it causes the better score.

**Evidence:** [Paper summary, section 4.3](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md). The figures show heatmaps for RTE and CB.

**How you can contribute:**
- **Measure it:** Define an "anchor score". For example:
  - the average attention **from** input tokens **to** the capsule, and
  - the average attention **from** the capsule **to** the input.

  Compute both over the **whole** dev set, for every layer and head.
- **Correlate:** Check whether examples with a higher anchor score are more often predicted correctly.
- **Intervene (causal test):** At test time, block attention to the capsule (set it to zero in the attention mask) and measure how much accuracy drops. Compare with blocking a random input token.
- Run this for both T5 and Llama (it links to Gap 12).

**A possible research question:** *"Is the attention anchor actually responsible for CaPT's gains?"*

**What you need:** 🟡 Medium. No new training is needed; you analyse trained models. Run with `output_attentions=True` to get the attention values.

---

#### Gap 18. Finding II was tested on one model and two tasks

**What it means:** The surprising result "adding one instance-aware token helps **without any training**" was only tested on T5-Base, on RTE and COPA.

**Why it matters:** COPA's dev set has 100 examples, so one example is one percentage point. A +4 point gain might partly be noise.

**How you can contribute:**
- Repeat the test on all 7 datasets, on T5-Large, Llama and Qwen, with several random seeds.
- Report averages **and** standard deviations.

**What you need:** 🟢 Easy. No training at all, only evaluation, so it is very cheap. A good warm-up project.

---

#### Gap 19. The statistics are fragile

**What it means:**
- CB (250 training examples) and COPA (400) are **tiny**. Results change a lot between random seeds.
- Each number is the average of only **3 runs**.
- The "test" set is the official **validation** set, because SuperGLUE test answers are hidden.
- MultiRC uses **sklearn macro-F1**, not the official SuperGLUE F1a, and exact match (EM) is not reported.

**Why it matters:** Some of the differences between methods (for example 77.51 vs 77.44) are smaller than the normal noise. They may not be real.

**Evidence:** [dataset.md, sections 2, 5 and 9](dataset.md#9-evaluation-metrics).

**How you can contribute:**
- Rerun CaPT and the baselines with **10 or more seeds**. Report the mean ± standard deviation, and use a **significance test** (for example a paired bootstrap test).
- Add the **official MultiRC F1a and EM** metrics.
- Submit to the official **SuperGLUE leaderboard** to get real test scores.

**A possible research question:** *"Are CaPT's reported gains statistically reliable?"* This is a **reproducibility study**, a well-respected type of paper.

**What you need:** 🟢 Easy but time-consuming. Many runs, but each one is small.

---

#### Gap 20. The learning rate still needs searching

**What it means:** CaPT says "no hyperparameter search needed" because there is no prompt-length search. But the learning rates are very different between models: **1** for T5, **7e-4** for Llama and **1e-1** for Qwen. Someone had to search for those.

**Why it matters:** If you must still search the learning rate for each new model, the "8× faster" claim is partly unfair. Only the prompt-length search was removed.

**Evidence:** [document.md, section 9](document.md#9-how-to-run-step-by-step) shows the three different `--lr` values.

**How you can contribute:**
- Do a **learning-rate sensitivity study**: train with 5–7 learning rates, and plot score against learning rate for CaPT and for the baselines. A flat curve means CaPT is robust.
- Try to find a **rule** for choosing the learning rate (for example based on hidden size), or a learning-rate-free optimizer.
- Recompute the training-time table with the learning-rate search counted for **all** methods.

**What you need:** 🟢 Easy.

---

#### Gap 21. Some comparisons are not fully fair

**What it means:**
- For Llama and Qwen, CaPT is compared with **"Linear Head"** (training only the classifier), not with full fine-tuning.
- CaPT on Llama and Qwen **also trains the classifier head**, so part of its gain may come from the head, not the capsule.
- Comparisons with **LoRA and adapters** are only in the appendix, not in the main table.

**Why it matters:** A fair comparison is needed to know how good CaPT really is for decoder-only models.

**How you can contribute:**
- Add **full fine-tuning** and **LoRA** results for Llama-3.2-1B on the same six tasks.
- Separate the effects: run (a) head only, (b) head + plain prompt-tuning, and (c) head + CaPT, all with the same settings.

**What you need:** 🟡 Medium. Full fine-tuning of a 1B model needs more GPU memory (about 24–40 GB).

---

#### Gap 22. LoPA is still more accurate

**What it means:** On T5-Base, LoPA scores **77.98** and CaPT scores **77.51**. CaPT wins on speed and size, but not on accuracy.

**Why it matters:** Some users care most about accuracy. Is there a method that is **both** as cheap as CaPT and as accurate as LoPA?

**How you can contribute:**
- Study **what LoPA does differently** (it combines a shared prompt with an instance-specific prompt through a low-rank network). Borrow its best idea in a cheap form.
- Combine ideas from Gaps 11, 15 and 16 and aim to beat 77.98 with fewer than 0.01% of the parameters.

**What you need:** 🟡 Medium.

---

#### Gap 23. Inference cost is not measured

**What it means:** The paper measures **training** time. At **inference** (prediction) time, CaPT computes a new capsule for every input at every layer, and the sequence is one token longer. Nobody measured how much slower or heavier this is.

**Why it matters:** For deployment, prediction speed and memory often matter more than training time.

**How you can contribute:**
- Measure **latency** (milliseconds per example), **throughput** (examples per second) and **GPU memory** at inference for: the frozen model, prompt-tuning, LoRA (merged and unmerged) and CaPT.
- Test different batch sizes and input lengths.

**What you need:** 🟢 Easy. No training needed. Use `torch.cuda.Event` timing, or Python's `time.perf_counter()` with `torch.cuda.synchronize()`.

---

#### Gap 24. Nobody knows what the capsule learns

**What it means:** The learned vector `p` is a list of 768 (or 2,048) numbers. The paper does not look at what it means or what information the capsule holds.

**Why it matters:** Understanding the capsule could explain why one vector is enough, and could lead to better designs.

**How you can contribute:**
- **Nearest words:** Find which word embeddings are closest to each learned `p`. This shows which "words" the task vector resembles.
- **Probing:** Train a small classifier on the capsule's hidden state to see whether it already holds the answer, and at which layer that appears.
- **Compare tasks:** Check whether capsules from similar tasks (RTE and CB, both entailment) are more similar to each other than capsules from different tasks.
- **Visualise:** Plot capsules from many inputs with t-SNE or UMAP and colour them by label.

**What you need:** 🟡 Medium. Mostly analysis, so little GPU time.

---

#### Gap 25. Input wording for decoder models is untested

**What it means:** The Llama and Qwen scripts define readable prompt templates such as `"Premise: …\nHypothesis: …\nAnswer:"`, but a code detail means these are **never used**. The decoder models see T5-style text instead (for example `rte sentence1: … sentence2: …`).

**Why it matters:** Decoder-only models were trained on natural text, so a natural template might help them, and could explain part of Gap 8.

**Evidence:** [dataset.md, section 6.3](dataset.md#63-how-llama--qwen-use-it) explains why `format_input()` skips the templates.

**How you can contribute:**
- Fix `format_input()` so the templates are applied, and compare: T5-style text against natural templates, for CaPT and the baselines.
- Check whether CaPT is **less sensitive** to wording than plain prompt-tuning. That would be a useful robustness result.

**What you need:** 🟢 Easy. A small code change plus a few runs on Llama-3.2-1B.

[Back to top](#table-of-contents)

---

## 5. How to Choose Your Project

Ask yourself three questions:

**1. How much GPU do you have?**

| Your hardware | Good gaps |
|---|---|
| No GPU / free Colab only | 18, 23, 24 (evaluation and analysis only) |
| One 16–24 GB GPU | 9, 10, 11, 13, 15, 17, 19, 20, 25 |
| One 40–80 GB GPU or cloud credits | 6, 7, 8, 12, 16, 21, 22 |

**2. How much time do you have?**

| Time | Suggestion |
|---|---|
| 2–4 weeks | One easy gap, such as 11 (masked pooling) or 18 (re-test Finding II). |
| 2–3 months | One medium gap, plus an easy one as a "bonus" experiment. |
| A thesis (6+ months) | Combine related gaps into one story (see section 6). |

**3. What kind of work do you enjoy?**

| You enjoy… | Good gaps |
|---|---|
| Writing new code / new methods | 11, 12, 13, 15, 16, 22 |
| Running careful experiments | 7, 9, 10, 19, 20, 21 |
| Understanding *why* things work | 14, 17, 24 |
| Practical, real-world use | 6, 23, 25 |

**Tip:** Gaps that **connect** to each other make a stronger story than one isolated experiment. For example, 8 + 12 + 17 forms one question: "Why is CaPT weaker on decoder-only models?"

[Back to top](#table-of-contents)

---

## 6. Five Ready-Made Project Ideas

Each idea below combines several gaps into one complete project.

### Idea 1. "Smarter Capsules": better pooling for CaPT 🟢
**Gaps:** 11, 15
**Question:** Can a better way of summarising the input improve CaPT while keeping it tiny?
**Plan:**
1. Fix padding with a masked mean (baseline improvement).
2. Add max, mean + max, attention pooling and a learned gate.
3. Test on T5-Base across all 6 SuperGLUE tasks, with 5 seeds.
4. Report a table of accuracy against trainable parameters.

**Good for:** A first research project, a course project or a workshop paper.

### Idea 2. "Anchors in Causal Models": fixing CaPT for decoder-only LLMs 🟡
**Gaps:** 8, 12, 17, 25
**Question:** Why is CaPT weaker on Llama and Qwen, and can moving or unmasking the capsule fix it?
**Plan:**
1. Measure the anchor score (Gap 17) for T5 and Llama.
2. Try capsule-at-end, a capsule that sees all tokens, and no `.detach()`.
3. Fix the prompt templates (Gap 25) as an extra factor.
4. Show which change closes the gap with T5.

**Good for:** A master's thesis or a conference paper.

### Idea 3. "Does One Vector Scale?": CaPT on large models and generation 🔴
**Gaps:** 6, 7, 13, 23
**Question:** Does a single capsule still work for 7B+ models and for text generation?
**Plan:**
1. Run CaPT with 4-bit frozen models at 1B, 3B and 8B.
2. Add a summarisation and a QA task.
3. Add decoder capsules for T5 (Gap 13).
4. Measure inference cost (Gap 23).

**Good for:** A larger thesis or a team project with cloud GPUs.

### Idea 4. "Is One Really All You Need?": a reproducibility study 🟢
**Gaps:** 14, 18, 19, 20
**Question:** Do CaPT's claims hold under stronger statistics?
**Plan:**
1. Rerun the main results with 10 seeds and significance tests.
2. Re-test Finding II widely.
3. Run a learning-rate sensitivity study.
4. Test the three hypotheses for why longer capsules hurt.

**Good for:** Reproducibility tracks (for example the ML Reproducibility Challenge). It needs little new code.

### Idea 5. "CaPT Meets LoRA": combining PEFT methods 🟡
**Gaps:** 16, 21, 22
**Question:** Does adding a 9K-parameter capsule on top of LoRA give a free accuracy boost?
**Plan:**
1. Add fair baselines (full fine-tuning, LoRA) for Llama.
2. Train LoRA + CaPT.
3. Try to beat LoPA's 77.98 on T5-Base with a combined method.

**Good for:** A practical paper or an engineering-focused project.

[Back to top](#table-of-contents)

---

## 7. Practical Tips for Starting

1. **Reproduce first.** Before changing anything, run the original CaPT on T5-Base + COPA and check that you get a score close to the paper's (see [document.md, section 9](document.md#9-how-to-run-step-by-step)). If you can't reproduce the baseline, your new results will not mean much.
2. **Change one thing at a time.** If you change the pooling **and** the position together, you won't know which one helped.
3. **Use several seeds.** Run at least 3, ideally 5, especially on CB and COPA. Always report mean ± standard deviation.
4. **Keep the comparison fair.** Use the same data splits, the same number of epochs and the same tuning effort for your method and for the baselines.
5. **Know where to edit:**

   | What you want to change | Where |
   |---|---|
   | How the capsule is built (pooling, combination) | `CaPT` class in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py#L664) / [modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py#L847) |
   | Where the capsule is placed (T5) | `T5Block.forward` in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py#L752) |
   | Where the capsule is placed (Llama) | `LlamaModel.forward` in [modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py#L950) |
   | New datasets | [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py), see [dataset.md, section 10](dataset.md#10-adding-a-new-dataset) |
   | Metrics | `score()` in [capt/train_t5.py](../CaPT/capt/train_t5.py#L403) |

6. **Reinstall after editing the library.** The `transformers/` folder is installed as a package. After you change a modeling file, run `python setup.py install` again, or use `pip install -e .` once so that your edits apply immediately.
7. **Check for newer work.** CaPT is a 2025 paper. Before you commit to a gap, search Google Scholar or Semantic Scholar for papers that cite it, so you don't repeat work someone has already done.
8. **Write down your question first.** One clear sentence, such as *"Does masked mean pooling improve CaPT on SuperGLUE?"*, keeps the whole project focused.

[Back to top](#table-of-contents)
