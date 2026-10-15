// Upload the self-contained demo page to Arweave (free < 100 KiB). usage: node upload_demo.mjs <jwk.json> build/demo.html
import fs from "fs";
import { TurboFactory } from "@ardrive/turbo-sdk";
const [jwkPath, p] = process.argv.slice(2);
const turbo = TurboFactory.authenticated({ privateKey: JSON.parse(fs.readFileSync(jwkPath, "utf8")) });
const buf = fs.readFileSync(p);
const res = await turbo.uploadFile({ fileStreamFactory: () => buf, fileSizeFactory: () => buf.length,
  dataItemOpts: { tags: [{ name: "Content-Type", value: "text/html" }, { name: "App-Name", value: "ModelProof" }, { name: "Type", value: "verifier-app" }] } });
console.log(JSON.stringify({ demo_tx: res.id }));
