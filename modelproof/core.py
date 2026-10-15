"""ModelProof core: a deterministic integer-only model format + reproducible inference.

Why integers: float inference differs across CPUs/GPUs/BLAS libraries, so "same model, same
input, same output" can't be checked bit-for-bit. ModelProof models use int8 weights, int32
biases and int64 accumulators with a fixed right-shift requantization. Every implementation
(Python here, JavaScript in the browser verifier) must produce byte-identical outputs, which
makes inference *verifiable* by anyone who can fetch the weights from Arweave.

Weights file (.mpw) layout, little-endian:
  magic   b"MPW1"
  u8      n_layers
  per layer:
    u16 in_dim, u16 out_dim, u8 shift, u8 relu(0/1)
    int8[out_dim*in_dim]  weights (row-major, row = output neuron)
    int32[out_dim]        biases
"""
import hashlib
import json
import struct

MAGIC = b"MPW1"


def encode_weights(layers):
    out = bytearray(MAGIC)
    out += struct.pack("<B", len(layers))
    for L in layers:
        out += struct.pack("<HHBB", L["in"], L["out"], L["shift"], 1 if L["relu"] else 0)
        out += struct.pack("<%db" % (L["in"] * L["out"]), *L["W"])
        out += struct.pack("<%di" % L["out"], *L["b"])
    return bytes(out)


def decode_weights(blob):
    if blob[:4] != MAGIC:
        raise ValueError("not a ModelProof weights file")
    p = 4
    (n,) = struct.unpack_from("<B", blob, p); p += 1
    layers = []
    for _ in range(n):
        i, o, shift, relu = struct.unpack_from("<HHBB", blob, p); p += 6
        W = list(struct.unpack_from("<%db" % (i * o), blob, p)); p += i * o
        b = list(struct.unpack_from("<%di" % o, blob, p)); p += 4 * o
        layers.append({"in": i, "out": o, "shift": shift, "relu": bool(relu), "W": W, "b": b})
    if p != len(blob):
        raise ValueError("trailing bytes in weights file")
    return layers


def clamp8(v):
    return -128 if v < -128 else 127 if v > 127 else v


def infer(layers, x):
    """x: list of int8 inputs. Returns final layer's int32 logits (no requant on last layer)."""
    h = list(x)
    for li, L in enumerate(layers):
        last = li == len(layers) - 1
        nxt = []
        for o in range(L["out"]):
            acc = L["b"][o]
            row = o * L["in"]
            for i in range(L["in"]):
                acc += L["W"][row + i] * h[i]
            if L["relu"] and acc < 0:
                acc = 0
            if not last:
                # arithmetic shift (floor), identical in JS via Math.floor(acc / 2**shift)
                acc = clamp8(acc >> L["shift"])
            nxt.append(acc)
        h = nxt
    return h


def argmax(v):
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def outputs_digest(outputs):
    """Canonical digest of a list of output vectors: sha256 of compact JSON."""
    s = json.dumps(outputs, separators=(",", ":"))
    return sha256_hex(s.encode())


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def verify(manifest, weights_blob):
    """Return (ok, report) after re-running every test vector in the manifest."""
    report = {}
    report["weights_sha256_expected"] = manifest["weights"]["sha256"]
    report["weights_sha256_actual"] = sha256_hex(weights_blob)
    report["weights_hash_ok"] = report["weights_sha256_expected"] == report["weights_sha256_actual"]
    report["weights_size_ok"] = manifest["weights"]["bytes"] == len(weights_blob)
    layers = decode_weights(weights_blob)
    outs = [infer(layers, v["input"]) for v in manifest["test_vectors"]]
    per = [o == v["output"] for o, v in zip(outs, manifest["test_vectors"])]
    report["vectors_total"] = len(per)
    report["vectors_matched"] = sum(per)
    report["outputs_digest_expected"] = manifest["outputs_digest"]
    report["outputs_digest_actual"] = outputs_digest(outs)
    ok = (report["weights_hash_ok"] and report["weights_size_ok"]
          and all(per) and report["outputs_digest_expected"] == report["outputs_digest_actual"])
    report["ok"] = ok
    return ok, report
