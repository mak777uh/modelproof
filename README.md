# ModelProof

**Permanent, verifiable AI models on Arweave.** ModelProof stores a model's weights and a signed-by-content manifest on the permaweb, and lets anyone — in Python, Node or a plain browser tab — re-run the model and prove that it produces exactly the outputs its publisher claimed.

Live demo (hosted on Arweave itself): https://arweave.net/cMlYaZOK54e63ATgWk21wtOdb25R3fZJdaFjqaz7fjA  
(mirror while the bundle settles: https://turbo-gateway.com/cMlYaZOK54e63ATgWk21wtOdb25R3fZJdaFjqaz7fjA)

## The problem
"Open" AI models are usually hosted on mutable, centralized storage. Files get replaced, repos get deleted, and there's no cheap way to check that the model you're running today is the model someone benchmarked yesterday. Even with the same file, float inference differs across CPUs/GPUs/libraries, so outputs can't be compared bit-for-bit.

## The approach
1. **Integer-only model format (MPW1).** int8 weights, int32 biases, integer accumulate, ReLU, arithmetic right-shift requantization. No floats at inference time → identical bytes on every platform.
2. **Manifest.** JSON with the weights' SHA-256 and size, the Arweave TX of the weights, preprocessing rules, metrics, a set of test vectors (input → exact output) and a digest of all outputs.
3. **Permanent storage.** Weights and manifest are uploaded to Arweave as ANS-104 data items through ArDrive Turbo (files under 100 KiB are free) and tagged `App-Name: ModelProof`.
4. **Verification.** Fetch manifest → fetch weights by TX → check hash → re-run every test vector → compare outputs digest. Two independent runtimes (Python and JavaScript) must agree.

## Live artifacts (Arweave)
| | TX |
|---|---|
| Manifest | `XCnP6FwdPqVPz5REcI7hHmBR8X75bKdOMGWXTOstbLI` |
| Verifier web app | `cMlYaZOK54e63ATgWk21wtOdb25R3fZJdaFjqaz7fjA` |
| Weights (117 bytes) | `SVHIFB9pR84SAv5TYwo_JUXJuAG8h7c1lTl-wmVmHl0` |

The sample model is a 4-8-3 MLP trained on Iris: 100% holdout accuracy in float and after int8 quantization (30 samples).

## Usage
```bash
python3 train_iris.py                                   # deterministic training + quantization -> build/
python3 cli.py verify build/manifest.draft.json --weights build/iris-mlp.mpw
npm install && node upload.mjs wallet.json build/iris-mlp.mpw build/manifest.draft.json
python3 cli.py verify XCnP6FwdPqVPz5REcI7hHmBR8X75bKdOMGWXTOstbLI   # fetches from Arweave
node verify_node.js XCnP6FwdPqVPz5REcI7hHmBR8X75bKdOMGWXTOstbLI       # independent JS runtime
python3 cli.py predict XCnP6FwdPqVPz5REcI7hHmBR8X75bKdOMGWXTOstbLI 6.3 3.3 6.0 2.5
```
No Python dependencies (stdlib only). Upload needs `@ardrive/turbo-sdk`.

## Roadmap
- Larger models: chunked weights with a Merkle root in the manifest.
- On-chain attestations: publish verification results as tagged Arweave items so a model accumulates independent "reproduced by" proofs.
- Run verifiers as Nosana GPU jobs for larger int8 models.

Built for the HackerNoon Decentralize AI Hackathon. MIT license.
