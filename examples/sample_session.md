# Sample Session — arXiv Digest Agent

All commands were run with `LLM_PROVIDER=ollama` (qwen2.5:7b-instruct). ANSI codes stripped.

---

### 1. Digest run

```
$ python -m agent.cli digest 2401.12345 --auto --no-qa --json-out .data/briefing_check.json
```

Output: PyMuPDF parsing degraded, trying pdfplumber fallback. Then the briefing is printed:

```
                   Distributionally Robust Receive Combining

Authors: Shixiong Wang, Wei Dai, Geoffrey Ye Li
Published: 2024-01-22T20:20:48Z
Categories: eess.SP
arXiv: 2401.12345 | PDF

-------------------------------------------------------------------------------

Why It Matters

This paper proposes a distributionally robust receive combining framework that
addresses uncertainties in wireless signal estimation, making it particularly
useful in challenging environments with spatially correlated signals and
arbitrary complex values. The method ensures robust performance by leveraging
norm regularization and empirical risk minimization, which are crucial for
handling uncertain covariance matrices.

...

Session saved. Thread ID: 2401.12345
Use 'python -m agent.cli ask 2401.12345 "your question"' to ask questions
```

---

### 2. In-paper ask with citations

```
$ python -m agent.cli ask 2401.12345 "What datasets did they evaluate on?" --verbose
```

Output:

```
node=start
node=qa_node

The method evaluated on datasets with pilot data and testing datasets. The
pilot dataset is used for training, while the testing dataset is used for
evaluation.

Citations:
  • experiments, p.?  (2401.12345:50)
  • experiments, p.?  (2401.12345:44)
```

The node transitions show only `start` and `qa_node` — no `fetch_pdf`, `parse`, `chunk_embed`, or `summarize` (the paper state is restored from the checkpointer).

---

### 3. Pronoun follow-up ask

```
$ python -m agent.cli ask 2401.12345 "How does it compare to the Wiener beamformer?" --verbose
```

Output:

```
node=start
node=qa_node

The Wiener beamformer and the Wiener-DL beamformer are compared in terms of
performance and computational burden. The Wiener-DL beamformer provides a good
balance between computational burden and performance, making it practically
promising. The Wiener-DR beamformer, although potentially better, has a
significant computational burden, which limits its practical use.

Citations:
  • experiments, p.?  (2401.12345:43)
  • Abstract, p.?  (2401.12345:28)
```

The pronoun "it" is rewritten using conversation history before retrieval.

---

### 4. Off-topic question returning abstain message

```
$ python -m agent.cli ask 2401.12345 "What does this paper say about the 2026 World Cup?" --verbose
```

Output:

```
node=start
node=qa_node

That isn't covered in this paper.
```

The abstain gate fires because the best retrieved chunk distance exceeds
`abstain_max_distance` (0.45). The LLM is not called. The exact abstain line
is: **That isn't covered in this paper.**

---

### 5. Sessions list

```
$ python -m agent.cli sessions
```

Output:

```
              Saved Sessions
┌─────────────────────────────────────────┐
│ Thread ID                               │
├─────────────────────────────────────────┤
│ 2401.12345                              │
│ topic:zzqxv-nonexistent-flibbertigibbet │
│ topic:zzqxv-qwzxq-flibberzzq            │
└─────────────────────────────────────────┘
```

`2401.12345` is listed as expected.
