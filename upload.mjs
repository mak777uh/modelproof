// Upload weights + manifest to Arweave via ArDrive Turbo (uploads < 100 KiB are free).
// usage: node upload.mjs <jwk.json> build/iris-mlp.mpw build/manifest.draft.json
// Writes build/manifest.json and build/arweave.json with the TX ids.
import fs from "fs";
import { TurboFactory } from "@ardrive/turbo-sdk";

const [jwkPath, wPath, mPath] = process.argv.slice(2);
const jwk = JSON.parse(fs.readFileSync(jwkPath, "utf8"));
const turbo = TurboFactory.authenticated({ privateKey: jwk });

async function up(buf, tags) {
  if (buf.length >= 100 * 1024) throw new Error("over free 100 KiB limit");
  const res = await turbo.uploadFile({
    fileStreamFactory: () => buf,
    fileSizeFactory: () => buf.length,
    dataItemOpts: { tags },
  });
  return res.id;
}

const weights = fs.readFileSync(wPath);
const manifest = JSON.parse(fs.readFileSync(mPath, "utf8"));
const common = [
  { name: "App-Name", value: "ModelProof" },
  { name: "App-Version", value: "0.1.0" },
  { name: "Model-Name", value: manifest.name },
];
const wId = await up(weights, [...common,
  { name: "Content-Type", value: "application/octet-stream" },
  { name: "Type", value: "model-weights" },
  { name: "Weights-SHA256", value: manifest.weights.sha256 }]);
manifest.weights.arweave_tx = wId;
const mBuf = Buffer.from(JSON.stringify(manifest, null, 1));
const mId = await up(mBuf, [...common,
  { name: "Content-Type", value: "application/json" },
  { name: "Type", value: "model-manifest" },
  { name: "Weights-TX", value: wId },
  { name: "Outputs-Digest", value: manifest.outputs_digest }]);
fs.writeFileSync("build/manifest.json", mBuf);
fs.writeFileSync("build/arweave.json", JSON.stringify({ weights_tx: wId, manifest_tx: mId }, null, 1));
console.log(JSON.stringify({ weights_tx: wId, manifest_tx: mId }));
