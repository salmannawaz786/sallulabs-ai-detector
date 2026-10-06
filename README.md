---
language: en
license: mit
tags: [ai-text-detection, text-classification, onnx, distillation, cpu]
pipeline_tag: text-classification
---

# SalluLabs AI Text Detector

A small, fast detector that tells human-written text from AI-generated text, built to run on an ordinary CPU.

| | |
|---|---|
| **99% accurate on essays** | 98.8% accuracy, 0.9995 AUROC on essays from topics it never saw in training; 96% on articles |
| **~30 ms per text** | 50 words on a single 2017 laptop CPU core, no GPU needed |
| **38 MB** | int8 ONNX model, needs only `onnxruntime`, `tokenizers` and `numpy` |
| **98.9 AUROC on RAID** | public leaderboard, 672k texts incl. adversarial attacks: catches 95.6% of AI text at a 5% false-positive rate |
| **Built for casual text too** | 97% accurate on social-media posts; only 6% false alarms on real casual chat messages |

> **Scope of the 99% figure:** it is measured on long-form writing (essays). Across all 11 held-out
> benchmarks below, including hard cases like paraphrased AI text and short chat replies, average
> accuracy is **89%** (0.954 AUROC). Results per text type are listed in full below.

## Quick start

```bash
pip install -r requirements.txt
python sallulabs_detector.py "Paste any text here to check it."
```

```python
from sallulabs_detector import Detector

det = Detector()              # loads ./model
det.predict("Some text...")
# {'label': 'ai', 'ai_probability': 0.97, 'high_confidence_ai': True, 'low_confidence': False, 'words': 42}
```

- `label`: `ai` or `human`, using a threshold calibrated on validation data.
- `high_confidence_ai`: past a stricter threshold set for about 1% false positives on validation data.
- `low_confidence`: under 25 words. Very short texts are unreliable for any detector.

Long texts are scored in overlapping 256-token windows, and the window scores are averaged.

## Accuracy

Every test set below was **held out from training**. They include a generator the model never saw (Cohere), essay topics it never saw, and AI text from a different dataset (MAGE).

| Test set (held out) | Texts | AUROC | Accuracy |
|---|---:|---:|---:|
| Student essays, unseen topics (DAIGT) | 3,000 | 0.9995 | **98.8%** |
| Social-media posts vs. AI posts on the same topic | 2,538 | 0.9964 | **97.4%** |
| Articles, reviews, abstracts, recipes… (RAID, unseen documents) | 3,000 | 0.9922 | **96.0%** |
| Text from an AI generator never seen in training (RAID, Cohere) | 1,411 | 0.9816 | **93.7%** |
| Short messages vs. short AI replies | 3,000 | 0.9743 | 92.1% |
| Conversation turns vs. AI turns | 2,784 | 0.9683 | 90.2% |
| Chat messages vs. AI chat replies | 348 | 0.9591 | 89.7% |
| Multi-domain benchmark (MAGE) | 3,000 | 0.9437 | 86.4% |
| GPT-4 text, different dataset (MAGE OOD) | 1,562 | 0.9601 | 84.7% |
| Instruction answers: human vs. AI, same prompt | 2,931 | 0.8794 | 79.3% |
| GPT-4 text, then paraphrased (MAGE OOD) | 2,362 | 0.8341 | 75.7% |
| **Average over all 11** | | **0.954** | **89.4%** |

**False alarms on real casual writing:** 3 of 50 hand-written casual chat messages, all human, were wrongly flagged as AI (6%).

### RAID leaderboard

<!-- RAID:START -->
Scored by the public [RAID leaderboard](https://raid-bench.xyz/leaderboard) on its hidden-label test set of 672,000 texts, covering 11 generators, 8 domains and 11 adversarial attacks ([evaluation](https://github.com/liamdugan/raid/pull/213)):

| RAID test set | AUROC | TPR @ 5% FPR | TPR @ 1% FPR |
|---|---:|---:|---:|
| **All texts, adversarial attacks included** | **98.94** | **95.61%** | **89.42%** |
| No adversarial attacks | 99.22 | 96.83% | 92.17% |

*TPR @ x% FPR: the share of AI text caught when only x% of human text is falsely flagged. The model was trained on RAID's training split; no test data was used.*
<!-- RAID:END -->

## Speed

Measured on an Intel Core i5 7th-gen laptop (2017, 2 cores), with onnxruntime on CPU:

| Text length | 1 thread | 2 threads |
|---|---:|---:|
| 50 words | 31 ms | 27 ms |
| 150 words | 110 ms | 80 ms |
| 400 words | 377 ms | 377 ms |

## How it was built

1. **Data:** over 300k training texts. They combine the RAID and MAGE benchmarks, DAIGT student essays, and pre-ChatGPT casual human writing (tweets, human-written dialogs, chat messages). Each human sample is paired with AI text written for the same prompt or conversation, from several open LLMs in randomized styles. This "mirror" pairing stops the model from using topic, length or tone as shortcuts.
2. **Teacher:** a 435M-parameter DeBERTa-v3-large classifier trained on that data.
3. **Student:** a 6-layer MiniLMv2 (~30M parameters) trained to match the teacher's predictions (knowledge distillation). It keeps most of the teacher's accuracy (0.954 vs. 0.967 average AUROC) at a fraction of the cost.
4. **Export:** ONNX with weight-only 8-bit quantization. It's verified against the full-precision model before release, including on short casual texts.

Input text is normalized before scoring. This removes zero-width characters, maps look-alike Cyrillic and Greek letters to Latin, and collapses whitespace, which undoes common detector-evasion tricks.

## Limitations

- **Short texts (<25 words) are unreliable.** That's true of every detector; results are flagged `low_confidence`.
- **Paraphrased AI text** (76%) and **long instruction-style answers** (79%) are the weakest areas.
- **The threshold is calibrated on the training mix.** On writing styles far from it, the false-positive rate can be higher. On MAGE's out-of-domain set, 28% of human texts were flagged at the default threshold.
- **English only.**
- **Don't use it as sole evidence** for academic or disciplinary decisions. A detector's score is a probability, not proof.

## License

MIT. See [LICENSE](LICENSE). The model is fine-tuned from [MiniLMv2](https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large) and distilled from a [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-large) teacher.
