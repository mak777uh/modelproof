// ModelProof JS runtime: independent re-implementation of MPW1 integer inference.
// Must match modelproof/core.py bit-for-bit. Works in browsers and Node 18+.
(function (root) {
  function decode(buf) {
    const dv = new DataView(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== "MPW1") throw new Error("not a ModelProof weights file");
    let p = 4;
    const n = dv.getUint8(p); p += 1;
    const layers = [];
    for (let l = 0; l < n; l++) {
      const inD = dv.getUint16(p, true), outD = dv.getUint16(p + 2, true);
      const shift = dv.getUint8(p + 4), relu = dv.getUint8(p + 5) === 1; p += 6;
      const W = new Array(inD * outD);
      for (let i = 0; i < W.length; i++) W[i] = dv.getInt8(p + i);
      p += W.length;
      const b = new Array(outD);
      for (let i = 0; i < outD; i++) b[i] = dv.getInt32(p + 4 * i, true);
      p += 4 * outD;
      layers.push({ in: inD, out: outD, shift, relu, W, b });
    }
    if (p !== dv.byteLength) throw new Error("trailing bytes in weights file");
    return layers;
  }
  const clamp8 = (v) => (v < -128 ? -128 : v > 127 ? 127 : v);
  function infer(layers, x) {
    let h = x.slice();
    layers.forEach((L, li) => {
      const last = li === layers.length - 1;
      const nxt = [];
      for (let o = 0; o < L.out; o++) {
        let acc = L.b[o];
        for (let i = 0; i < L.in; i++) acc += L.W[o * L.in + i] * h[i];
        if (L.relu && acc < 0) acc = 0;
        if (!last) acc = clamp8(Math.floor(acc / Math.pow(2, L.shift)));
        nxt.push(acc);
      }
      h = nxt;
    });
    return h;
  }
  async function sha256Hex(bytes) {
    const subtle = (globalThis.crypto && globalThis.crypto.subtle) || require("crypto").webcrypto.subtle;
    const d = await subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  // Same canonical form as Python json.dumps(outputs, separators=(",",":"))
  const outputsDigest = (outs) => sha256Hex(new TextEncoder().encode(JSON.stringify(outs)));
  async function verify(manifest, bytes) {
    const r = {};
    r.weights_sha256_expected = manifest.weights.sha256;
    r.weights_sha256_actual = await sha256Hex(bytes);
    r.weights_hash_ok = r.weights_sha256_expected === r.weights_sha256_actual;
    r.weights_size_ok = manifest.weights.bytes === bytes.byteLength;
    const layers = decode(bytes);
    const outs = manifest.test_vectors.map((v) => infer(layers, v.input));
    const per = outs.map((o, i) => JSON.stringify(o) === JSON.stringify(manifest.test_vectors[i].output));
    r.vectors_total = per.length;
    r.vectors_matched = per.filter(Boolean).length;
    r.outputs_digest_expected = manifest.outputs_digest;
    r.outputs_digest_actual = await outputsDigest(outs);
    r.ok = r.weights_hash_ok && r.weights_size_ok && per.every(Boolean) &&
      r.outputs_digest_expected === r.outputs_digest_actual;
    return r;
  }
  function preprocess(manifest, vals) {
    const pp = manifest.preprocessing;
    // Python round() is banker's rounding; match it exactly.
    const pyRound = (v) => { const f = Math.floor(v), d = v - f; if (d > 0.5) return f + 1; if (d < 0.5) return f; return f % 2 === 0 ? f : f + 1; };
    return vals.map((v, i) => clamp8(pyRound(((v - pp.mean[i]) / pp.std[i]) * pp.input_scale)));
  }
  const api = { decode, infer, sha256Hex, outputsDigest, verify, preprocess };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ModelProof = api;
})(this);
