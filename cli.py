"""ModelProof CLI.

  python3 cli.py verify <manifest.json|ARWEAVE_TX> [--weights FILE]
      Re-runs every test vector and checks the weights hash. With an Arweave TX id it fetches
      the manifest and the weights from the gateway, so anyone can audit the model.
  python3 cli.py predict <manifest.json|TX> sl sw pl pw
      Runs reproducible inference for one Iris sample (cm).
"""
import json
import sys
import urllib.request

from modelproof.core import verify, decode_weights, infer, argmax

GATEWAYS = ["https://arweave.net/", "https://turbo-gateway.com/", "https://ardrive.net/"]


def fetch(ref):
    if ref.endswith(".json") or "/" in ref:
        return open(ref, "rb").read()
    last = None
    for g in GATEWAYS:  # fall back across gateways (fresh bundles appear on turbo first)
        try:
            with urllib.request.urlopen(g + ref, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise SystemExit("could not fetch %s from any gateway: %s" % (ref, last))


def load(ref, weights_path=None):
    manifest = json.loads(fetch(ref))
    if weights_path:
        blob = open(weights_path, "rb").read()
    else:
        tx = manifest["weights"].get("arweave_tx")
        if not tx:
            raise SystemExit("manifest has no arweave_tx; pass --weights FILE")
        blob = fetch(tx)
    return manifest, blob


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    cmd, ref = argv[0], argv[1]
    wpath = argv[argv.index("--weights") + 1] if "--weights" in argv else None
    manifest, blob = load(ref, wpath)
    if cmd == "verify":
        ok, rep = verify(manifest, blob)
        print(json.dumps(rep, indent=1))
        print("VERIFIED" if ok else "MISMATCH")
        return 0 if ok else 1
    if cmd == "predict":
        vals = [float(v) for v in argv[2:6]]
        pp = manifest["preprocessing"]
        s = pp["input_scale"]
        x = [max(-128, min(127, round((v - m) / d * s))) for v, m, d in zip(vals, pp["mean"], pp["std"])]
        out = infer(decode_weights(blob), x)
        print(json.dumps({"input_int8": x, "logits": out,
                          "class": manifest["task"]["classes"][argmax(out)]}))
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
