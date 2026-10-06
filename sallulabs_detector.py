"""SalluLabs AI-text detector: CPU inference for the int8 ONNX model in ./model.

Needs only onnxruntime, tokenizers and numpy.

    from sallulabs_detector import Detector
    det = Detector()
    print(det.predict("Some text to check"))

Command line:
    python sallulabs_detector.py "Some text to check"
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent / "model"
MAX_LEN, STRIDE, MAX_CHUNKS = 256, 32, 8
MIN_WORDS = 25  # below this, verdicts are flagged low-confidence

# ── text normalization (undoes homoglyph / zero-width / whitespace tricks) ──
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "ս": "u",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ν": "v", "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
})
_URL = re.compile(r"https?://\S+")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\s*\n\s*")


def normalize(text):
    t = unicodedata.normalize("NFKC", text if isinstance(text, str) else "")
    t = t.translate(_ZERO_WIDTH).translate(_HOMOGLYPHS)
    t = _URL.sub("", t)
    t = _SPACES.sub(" ", t)
    return _NEWLINES.sub("\n", t).strip()


class Detector:
    def __init__(self, model_dir=MODEL_DIR, threads=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) // 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(d / "model_int8.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=MAX_LEN, stride=STRIDE)
        self.pad_id = self.tok.token_to_id("<pad>") or 0
        th = json.loads((d / "thresholds.json").read_text())
        self.threshold, self.strict = th["balanced"], th["strict_1pct_fpr"]

    def _margins(self, texts, batch_size=64):
        """AI-minus-human logit margin per text: overlapping 256-token windows, length-weighted mean."""
        ids, owner = [], []
        for i, enc in enumerate(self.tok.encode_batch([normalize(t) for t in texts])):
            for piece in ([enc] + list(enc.overflowing))[:MAX_CHUNKS]:
                ids.append(piece.ids)
                owner.append(i)
        owner = np.asarray(owner)
        lengths = np.array([len(x) for x in ids], dtype=np.float64)
        out = np.zeros(len(ids))
        order = np.argsort(lengths)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            n = int(lengths[idx].max())
            input_ids = np.full((len(idx), n), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), n), dtype=np.int64)
            for r, j in enumerate(idx):
                input_ids[r, :len(ids[j])] = ids[j]
                mask[r, :len(ids[j])] = 1
            logits = self.sess.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]
            out[idx] = logits[:, 1] - logits[:, 0]
        num = np.bincount(owner, weights=out * lengths, minlength=len(texts))
        return num / np.maximum(np.bincount(owner, weights=lengths, minlength=len(texts)), 1e-9)

    def predict(self, texts):
        """Returns one dict per text: label ('ai' / 'human'), ai_probability, confidence flags."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        results = []
        for t, m in zip(texts, self._margins(texts)):
            words = len(t.split())
            results.append({
                "label": "ai" if m >= self.threshold else "human",
                "ai_probability": float(1 / (1 + np.exp(-(m - self.threshold)))),
                "high_confidence_ai": bool(m >= self.strict),
                "low_confidence": words < MIN_WORDS,
                "words": words,
            })
        return results[0] if single else results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    r = Detector().predict(text)
    print(f"{'AI-generated' if r['label'] == 'ai' else 'Human-written'} "
          f"(AI probability {r['ai_probability'] * 100:.1f}%, {r['words']} words)"
          + ("  [short text: low confidence]" if r["low_confidence"] else ""))
