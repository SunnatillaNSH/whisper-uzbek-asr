/**
 * RunPod Serverless endpoint'ini Cloudflare Worker'dan chaqirish misoli.
 *
 * Sozlash (wrangler.toml / Dashboard → Settings → Variables):
 *   RUNPOD_API_KEY      — maxfiy (Secret sifatida saqlang, oddiy var emas)
 *   RUNPOD_ENDPOINT_ID  — masalan "vwobkifazoyyh1"
 *
 * Ikki usul qo'llab-quvvatlanadi:
 *   transcribeFromUrl()   — audio allaqachon R2/S3 da bo'lsa (TAVSIYA ETILADI)
 *   transcribeFromBytes() — audio Worker xotirasida bo'lsa
 */

const RUNPOD_BASE = "https://api.runpod.ai/v2";

/**
 * Uzun audio uchun asinxron usul: /run bilan yuborib, /status orqali kuzatamiz.
 * Worker'ning CPU vaqti cheklangani uchun uzoq kutish o'rniga job ID qaytarib,
 * keyin alohida so'rov bilan natijani olish ham mumkin.
 */
async function submitJob(env, input) {
  const resp = await fetch(`${RUNPOD_BASE}/${env.RUNPOD_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
  });
  if (!resp.ok) throw new Error(`RunPod /run xatosi: ${resp.status}`);
  return resp.json(); // { id, status: "IN_QUEUE" }
}

async function getJobStatus(env, jobId) {
  const resp = await fetch(
    `${RUNPOD_BASE}/${env.RUNPOD_ENDPOINT_ID}/status/${jobId}`,
    { headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` } },
  );
  if (!resp.ok) throw new Error(`RunPod /status xatosi: ${resp.status}`);
  return resp.json();
}

/**
 * Qisqa audio uchun sinxron usul: javobni kutib turadi.
 * DIQQAT: sovuq start + uzun audio bo'lsa, Worker'ning so'rov vaqti chegarasiga
 * urilishi mumkin. 1-2 daqiqadan uzun audio uchun submitJob() ni ishlating.
 */
async function transcribeSync(env, input) {
  const resp = await fetch(`${RUNPOD_BASE}/${env.RUNPOD_ENDPOINT_ID}/runsync`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
  });
  if (!resp.ok) throw new Error(`RunPod /runsync xatosi: ${resp.status}`);

  const data = await resp.json();
  const out = data.output || {};
  if (out.status !== "success") {
    throw new Error(`Transkripsiya xatosi: ${out.message || JSON.stringify(data)}`);
  }
  return out; // { status, text, duration_sec, processing_time_sec }
}

/** Audio R2/S3 da bo'lsa — eng tejamkor yo'l, fayl Worker orqali o'tmaydi. */
export async function transcribeFromUrl(env, audioUrl) {
  return transcribeSync(env, { audio_url: audioUrl });
}

/** Audio baytlari Worker qo'lida bo'lsa (masalan so'rov tanasidan kelgan). */
export async function transcribeFromBytes(env, arrayBuffer) {
  // ArrayBuffer → base64 (katta fayllarda xotirani tejash uchun bo'lak-bo'lak)
  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return transcribeSync(env, { audio_base64: btoa(binary) });
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("POST kutilmoqda", { status: 405 });
    }

    try {
      const { audioUrl, r2Key } = await request.json();

      // 1-variant: to'g'ridan-to'g'ri havola
      if (audioUrl) {
        const result = await transcribeFromUrl(env, audioUrl);
        return Response.json(result);
      }

      // 2-variant: R2 dagi obyekt (Worker o'qib, base64 sifatida yuboradi)
      if (r2Key && env.BUCKET) {
        const obj = await env.BUCKET.get(r2Key);
        if (!obj) return new Response("R2 da topilmadi", { status: 404 });
        const result = await transcribeFromBytes(env, await obj.arrayBuffer());
        return Response.json(result);
      }

      return new Response("audioUrl yoki r2Key berilishi shart", { status: 400 });
    } catch (err) {
      return Response.json({ status: "error", message: String(err) }, { status: 500 });
    }
  },
};
