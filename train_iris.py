"""Train a tiny MLP on Iris (pure Python, fixed seed), quantize to the integer ModelProof
format, and write weights + manifest. Deterministic: same seed -> same bytes.

usage: python3 train_iris.py  -> build/iris-mlp.mpw, build/manifest.draft.json
"""
import csv
import json
import math
import os
import random

from modelproof.core import encode_weights, infer, argmax, sha256_hex, outputs_digest

random.seed(1337)
CLASSES = ["setosa", "versicolor", "virginica"]
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "iris.csv"))))
X = [[float(r[k]) for k in ("sepal_length", "sepal_width", "petal_length", "petal_width")] for r in rows]
Y = [CLASSES.index(r["species"]) for r in rows]
mean = [sum(c) / len(c) for c in zip(*X)]
std = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) for c, m in zip(zip(*X), mean)]
Xn = [[(v - m) / s for v, m, s in zip(x, mean, std)] for x in X]

idx = list(range(len(X)))
random.shuffle(idx)
test_idx = sorted(idx[:30])
train_idx = idx[30:]

H = 8
W1 = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(H)]
b1 = [0.0] * H
W2 = [[random.gauss(0, 0.5) for _ in range(H)] for _ in range(3)]
b2 = [0.0] * 3


def fwd(x):
    z1 = [sum(w * v for w, v in zip(W1[j], x)) + b1[j] for j in range(H)]
    h = [max(0.0, z) for z in z1]
    z2 = [sum(w * v for w, v in zip(W2[k], h)) + b2[k] for k in range(3)]
    m = max(z2)
    e = [math.exp(z - m) for z in z2]
    s = sum(e)
    return z1, h, [v / s for v in e]


lr = 0.05
for epoch in range(300):
    random.shuffle(train_idx)
    for i in train_idx:
        x, y = Xn[i], Y[i]
        z1, h, p = fwd(x)
        d2 = [p[k] - (1.0 if k == y else 0.0) for k in range(3)]
        d1 = [(sum(W2[k][j] * d2[k] for k in range(3)) if z1[j] > 0 else 0.0) for j in range(H)]
        for k in range(3):
            for j in range(H):
                W2[k][j] -= lr * d2[k] * h[j]
            b2[k] -= lr * d2[k]
        for j in range(H):
            for t in range(4):
                W1[j][t] -= lr * d1[j] * x[t]
            b1[j] -= lr * d1[j]

# ---- quantization ----
SX = 32.0  # input scale: int8 = round(normalized * 32)


def qin(xn):
    return [max(-128, min(127, round(v * SX))) for v in xn]


s1 = 127.0 / max(abs(w) for r in W1 for w in r)
W1q = [max(-128, min(127, round(w * s1))) for r in W1 for w in r]
b1q = [round(b * SX * s1) for b in b1]
max_h = max(max(fwd(x)[1]) for x in Xn)
shift = 0
while (SX * s1 / (2 ** shift)) * max_h > 127:
    shift += 1
sh = SX * s1 / (2 ** shift)
s2 = 127.0 / max(abs(w) for r in W2 for w in r)
W2q = [max(-128, min(127, round(w * s2))) for r in W2 for w in r]
b2q = [round(b * sh * s2) for b in b2]

layers = [
    {"in": 4, "out": H, "shift": shift, "relu": True, "W": W1q, "b": b1q},
    {"in": H, "out": 3, "shift": 0, "relu": False, "W": W2q, "b": b2q},
]
blob = encode_weights(layers)


def acc(ids, quant):
    ok = 0
    for i in ids:
        pred = argmax(infer(layers, qin(Xn[i]))) if quant else argmax(fwd(Xn[i])[2])
        ok += pred == Y[i]
    return ok / len(ids)


float_acc, q_acc = acc(test_idx, False), acc(test_idx, True)
vectors = []
for i in test_idx:
    xi = qin(Xn[i])
    vectors.append({"input": xi, "output": infer(layers, xi), "label": CLASSES[Y[i]]})

os.makedirs("build", exist_ok=True)
open("build/iris-mlp.mpw", "wb").write(blob)
manifest = {
    "schema": "modelproof/manifest@1",
    "name": "iris-mlp-int8",
    "version": "0.1.0",
    "description": "4-8-3 MLP trained on Iris, int8 weights, integer-only reproducible inference",
    "license": "MIT",
    "task": {"type": "classification", "classes": CLASSES},
    "architecture": "dense(4->8,relu,>>%d) -> dense(8->3)" % shift,
    "preprocessing": {"features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                      "mean": [round(m, 6) for m in mean], "std": [round(s, 6) for s in std],
                      "input_scale": SX, "rule": "int8 = clamp(round((x-mean)/std*input_scale),-128,127)"},
    "runtime": {"spec": "MPW1 integer inference: int64 accumulate, relu, arithmetic >>shift, clamp int8; last layer raw int32 logits"},
    "weights": {"format": "MPW1", "bytes": len(blob), "sha256": sha256_hex(blob), "arweave_tx": None},
    "metrics": {"holdout_n": len(test_idx), "float_accuracy": float_acc, "int8_accuracy": q_acc},
    "training": {"seed": 1337, "epochs": 300, "lr": lr, "source": "train_iris.py"},
    "test_vectors": vectors,
    "outputs_digest": outputs_digest([v["output"] for v in vectors]),
}
json.dump(manifest, open("build/manifest.draft.json", "w"), indent=1)
print("weights", len(blob), "bytes sha256", manifest["weights"]["sha256"])
print("float acc", float_acc, "int8 acc", q_acc, "shift", shift)
