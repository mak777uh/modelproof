// Generic free (<100 KiB) Arweave upload. usage: node upload_file.mjs <jwk.json> <file> <content-type> [Type-tag]
import fs from "fs";
import { TurboFactory } from "@ardrive/turbo-sdk";
const [jwkPath, p, ct, type] = process.argv.slice(2);
const turbo = TurboFactory.authenticated({ privateKey: JSON.parse(fs.readFileSync(jwkPath, "utf8")) });
const buf = fs.readFileSync(p);
if (buf.length >= 100 * 1024) throw new Error("over free 100 KiB limit");
const res = await turbo.uploadFile({ fileStreamFactory: () => buf, fileSizeFactory: () => buf.length,
  dataItemOpts: { tags: [{ name: "Content-Type", value: ct }, { name: "App-Name", value: "ModelProof" }, { name: "Type", value: type || "file" }] } });
console.log(JSON.stringify({ id: res.id, bytes: buf.length }));
