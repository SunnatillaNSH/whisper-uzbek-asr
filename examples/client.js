// Node.js 18+ / Cloudflare Workers mijoz
// Foydalanish: API_URL=https://xxxx.ngrok-free.app node examples/client.js audio.wav
import { readFile } from "node:fs/promises";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

export async function transcribe(bytes, filename = "audio.wav", mime = "audio/wav") {
  const form = new FormData();
  form.append("file", new Blob([bytes], { type: mime }), filename);

  const res = await fetch(`${API_URL}/transcribe`, {
    method: "POST",
    headers: { "ngrok-skip-browser-warning": "true" },
    body: form,
  });
  const data = await res.json();
  if (data.status !== "success") throw new Error(`API xatosi (${res.status}): ${data.message}`);
  return data.text;
}

// Cloudflare Worker ichida R2'dan olingan obyekt bilan:
//   const bytes = await env.BUCKET.get(key).then(o => o.arrayBuffer());
//   const text = await transcribe(bytes, key, "audio/mp3");

if (process.argv[2]) {
  const bytes = await readFile(process.argv[2]);
  console.log(await transcribe(bytes, process.argv[2].split("/").pop()));
}
