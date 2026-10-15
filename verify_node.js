// Cross-implementation check: verify a manifest with the JS runtime (Node).
// usage: node verify_node.js <manifest.json|ARWEAVE_TX> [weights.mpw]
const fs = require("fs");
const MP = require("./web/modelproof.js");
const GWS = ["https://arweave.net/", "https://turbo-gateway.com/", "https://ardrive.net/"];
async function get(ref) {
  if (fs.existsSync(ref)) return new Uint8Array(fs.readFileSync(ref));
  for (const g of GWS) {
    try { const r = await fetch(g + ref); if (r.ok) return new Uint8Array(await r.arrayBuffer()); } catch (e) {}
  }
  throw new Error("could not fetch " + ref + " from any gateway");
}
(async () => {
  const [ref, wref] = process.argv.slice(2);
  const manifest = JSON.parse(new TextDecoder().decode(await get(ref)));
  const bytes = await get(wref || manifest.weights.arweave_tx);
  const rep = await MP.verify(manifest, bytes);
  console.log(JSON.stringify(rep, null, 1));
  console.log(rep.ok ? "VERIFIED (js)" : "MISMATCH (js)");
  process.exit(rep.ok ? 0 : 1);
})();
