/**
 * Ovoz Konsoli — RunPod proksisi, jonli oqim va statistika.
 *
 * NEGA PROKSI. Konsol sahifasi internetda ochiq. RunPod API kaliti sahifa
 * kodiga yozilsa, havolani topgan har kim uni "View Source" orqali o'qiy
 * oladi — kalit esa butun hisobga kirish huquqini beradi (pod yaratish,
 * o'chirish, pul sarflash). Shuning uchun kalit Cloudflare secret'ida yotadi
 * va brauzerga HECH QACHON yuborilmaydi.
 *
 * Faqat uchta yo'l ochiq: health, run, status. Pod yaratish yoki o'chirish
 * kabi RunPod API'lariga proksi orqali umuman yetib bo'lmaydi.
 *
 * Secret'lar:
 *   npx wrangler secret put RUNPOD_API_KEY --name ovoz-konsoli   (majburiy)
 *   npx wrangler secret put FEED_TOKEN     --name ovoz-konsoli   (oqim uchun)
 *   npx wrangler secret put API_TOKENS     --name ovoz-konsoli   (mijoz kalitlari)
 *
 * Deploy (config yo'lini ANIQ bering — aks holda wrangler boshqa konfigni
 * topib, noto'g'ri nom bilan Worker yaratadi):
 *   npx wrangler deploy --config console/wrangler.toml
 */

const ENDPOINT = "vwobkifazoyyh1";
const BASE = `https://api.runpod.ai/v2/${ENDPOINT}`;
const TTL = 7200;                      // 2 soat — batafsil oqim (audio + matn)
const STAT_TTL = 60 * 60 * 24 * 120;   // 120 kun — kunlik statistika
const MAX_STORE = 20 * 1024 * 1024;    // KV chegarasi 25 MB, zaxira bilan

const json = (d, status = 200) =>
  Response.json(d, { status, headers: { "cache-control": "no-store" } });

/**
 * Transkripsiya yo'llarini himoyalash — UCH HOLATLI.
 *
 * Nega uchta holat kerak: himoyani yoqish ikki tomonni bir vaqtda
 * o'zgartirishni talab qiladi (proksi tekshira boshlaydi, SellUp token
 * yubora boshlaydi). Qaysi biri oldin deploy qilinsa, o'sha oraliqda
 * production sinadi. Kuzatuv rejimi shu oraliqni yo'q qiladi.
 *
 *   API_TOKENS yo'q              -> ochiq, hech narsa tekshirilmaydi
 *   API_TOKENS + API_ENFORCE!=1  -> KUZATUV: hamma o'tadi, lekin tokensiz
 *                                   so'rovlar kunlik hisoblagichga tushadi
 *   API_TOKENS + API_ENFORCE=1   -> yoqilgan: tokensiz so'rov rad etiladi
 *
 * Tartib: kuzatuv -> SellUp tomonida token -> statistikada no_token NOL
 * bo'lgani tasdiqlanadi -> yoqish.
 *
 * FEED_TOKEN ham qabul qilinadi: konsolning o'zi shu bilan ishlaydi.
 */
function checkApi(req, env) {
  if (!env.API_TOKENS) return { ok: true, enforced: false, gated: false };

  const allowed = env.API_TOKENS.split(",").map((s) => s.trim()).filter(Boolean);
  if (env.FEED_TOKEN) allowed.push(env.FEED_TOKEN);

  const t = (req.headers.get("authorization") || "").replace(/^Bearer\s+/i, "").trim()
    || req.headers.get("x-api-token") || "";

  return { ok: t.length > 0 && allowed.includes(t), enforced: env.API_ENFORCE === "1", gated: true };
}

/** Oqim ochiq bo'lmasligi kerak — unda real mijoz suhbatlari bor. */
function feedAllowed(req, env) {
  if (!env.FEED_TOKEN) return false;
  const t = req.headers.get("x-feed-token") || new URL(req.url).searchParams.get("t");
  return t === env.FEED_TOKEN;
}

/**
 * Audio formatini BAYTLARDAN aniqlaydi.
 *
 * Brauzer pleyeri content-type'ga qarab dekoder tanlaydi. Hamma faylni
 * "audio/mpeg" deb bersak, WAV yoki M4A noto'g'ri dekoder bilan ochiladi —
 * pleyer davomiylikni noto'g'ri ko'rsatadi yoki yarmida to'xtaydi.
 */
function sniffType(b) {
  const ascii = (i, n) => String.fromCharCode(...b.slice(i, i + n));
  if (b.length < 12) return "application/octet-stream";
  if (ascii(0, 4) === "RIFF" && ascii(8, 4) === "WAVE") return "audio/wav";
  if (ascii(0, 4) === "fLaC") return "audio/flac";
  if (ascii(0, 4) === "OggS") return "audio/ogg";
  if (ascii(4, 4) === "ftyp") return "audio/mp4";
  if (ascii(0, 3) === "ID3") return "audio/mpeg";
  if (b[0] === 0xff && (b[1] & 0xe0) === 0xe0) {
    return (b[1] & 0x16) === 0x10 ? "audio/aac" : "audio/mpeg";
  }
  return "application/octet-stream";
}

async function storeBytes(env, id, bin) {
  if (bin.byteLength > MAX_STORE) return { kind: "too_big", bytes: bin.byteLength };
  const type = sniffType(bin);
  await env.OVOZ_FEED.put(`audio:${id}`, bin, { expirationTtl: TTL, metadata: { type } });
  return { kind: "stored", bytes: bin.byteLength, type };
}

async function saveEntry(env, id, patch) {
  const key = `job:${id}`;
  const prev = (await env.OVOZ_FEED.get(key, "json")) || {};
  const next = { ...prev, ...patch, id };
  await env.OVOZ_FEED.put(key, JSON.stringify(next), { expirationTtl: TTL });
  return next;
}

/**
 * Audioni oqim uchun saqlaydi.
 *
 * `audio_url` bo'lsa ham baytlarni YUKLAB OLAMIZ. Faqat havolani saqlash
 * yetarli emas edi: production aynan shu ko'rinishda yuboradi, havola esa
 * brauzerdan ochilmasligi mumkin (autentifikatsiya, muddat, CORS) — natijada
 * oqimda matn bor, audio yo'q bo'lib qolardi. Solishtirish esa aynan shu
 * ikkisi yonma-yon turganda ma'noga ega.
 *
 * Yuklab olish javobni kechiktirmasligi uchun ctx.waitUntil orqali fonda.
 */
async function saveAudio(env, ctx, id, input) {
  try {
    if (input.audio_base64) {
      return await storeBytes(env, id, Uint8Array.from(atob(input.audio_base64), (c) => c.charCodeAt(0)));
    }
    if (input.audio_url) {
      const url = input.audio_url;
      ctx.waitUntil((async () => {
        try {
          const r = await fetch(url, { headers: { "User-Agent": "ovoz-konsoli/1.0" }, redirect: "follow" });
          if (!r.ok) return;
          const res = await storeBytes(env, id, new Uint8Array(await r.arrayBuffer()));
          await saveEntry(env, id, { audio: { ...res, url } });
        } catch { /* audio bo'lmasa ham matn qoladi */ }
      })());
      return { kind: "url", url };
    }
  } catch { /* saqlay olmasak ham transkripsiya davom etsin */ }
  return { kind: "none" };
}

async function bumpDay(env, patch) {
  const day = new Date().toISOString().slice(0, 10);
  const key = `stat:${day}`;
  const cur = (await env.OVOZ_FEED.get(key, "json"))
    || { day, count: 0, errors: 0, audio_sec: 0, proc_sec: 0, queue_sec: 0 };
  patch(cur);
  await env.OVOZ_FEED.put(key, JSON.stringify(cur), { expirationTtl: STAT_TTL });
}

/**
 * Kunlik statistika. Batafsil oqim 2 soatdan keyin o'chadi, lekin "qancha
 * so'rov bo'ldi, qancha audio, qancha GPU vaqti" degan savollar keyin ham
 * kerak. Har bir job FAQAT BIR MARTA sanaladi: /status bir necha marta
 * so'raladi, shuning uchun yozuvga `counted` bayrog'i qo'yiladi.
 */
const bumpStats = (env, entry, out, failed) => bumpDay(env, (c) => {
  c.count += 1;
  if (failed) c.errors += 1;
  c.audio_sec += out.duration_sec || 0;
  c.proc_sec += out.processing_time_sec || 0;
  c.queue_sec += entry.queue_sec || 0;
});

/**
 * Kuzatuv rejimida tokensiz kelgan so'rovlarni MANBA BO'YICHA sanaydi.
 *
 * Nega yagona raqam yetarli emas: kuzatuv paytida o'z tahlil skriptlarimiz
 * ham tokensiz ketadi va hisoblagichni oshiradi. Yagona son bo'lsa,
 * "SellUp hali token yubormayaptimi yoki bu bizning skriptmi" — ajratib
 * bo'lmaydi va API_ENFORCE=1 ni yoqish xavfli bo'lib qoladi.
 *
 * Manba User-Agent bo'yicha guruhlanadi. Kardinallik cheklangan: 12 tadan
 * ortiq turli manba bo'lsa, qolgani "boshqa" ga yig'iladi — aks holda KV
 * yozuvi cheksiz o'sib ketadi.
 */
function sourceKey(req) {
  const ua = (req.headers.get("user-agent") || "").trim();
  if (!ua) return "(ua yo'q)";
  return ua.slice(0, 48);
}

function addSource(bucket, req) {
  const k = sourceKey(req);
  if (bucket[k] === undefined && Object.keys(bucket).length >= 12) {
    bucket["boshqa"] = (bucket["boshqa"] || 0) + 1;
  } else {
    bucket[k] = (bucket[k] || 0) + 1;
  }
}

/**
 * Kuzatuv hisoblagichlari — FAQAT /run uchun.
 *
 * Nega faqat /run: har bir job o'ndan ortiq marta /status bilan so'raladi
 * va konsol /health ni har 10 soniyada chaqiradi. Ularning har birida KV'ga
 * yozsak, kunlik yozuv chegarasi tez tugaydi. Mijoz esa har uchala yo'lga
 * bir xil sarlavha yuboradi, shuning uchun /run ni sanash yetarli.
 *
 * Ikki tomonlama: no_token_by — tokensiz kelganlar, auth_ok_by — token
 * bilan kelganlar. Ikkinchisi muhim, chunki "tokensiz hech kim kelmadi"
 * degani "production token bilan kelyapti" degani EMAS — trafik umuman
 * bo'lmagan bo'lishi ham mumkin. auth_ok_by ijobiy tasdiq beradi.
 */
const bumpAuth = (env, req, ok) => bumpDay(env, (c) => {
  if (ok) {
    addSource(c.auth_ok_by || (c.auth_ok_by = {}), req);
  } else {
    c.no_token = (c.no_token || 0) + 1;
    addSource(c.no_token_by || (c.no_token_by = {}), req);
  }
});

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (!url.pathname.startsWith("/api/")) return env.ASSETS.fetch(req);

    const [, , action, id] = url.pathname.split("/");

    // ---------- jonli oqim (FEED_TOKEN bilan) ----------
    if (action === "feed") {
      if (!feedAllowed(req, env)) return json({ error: "Oqim yopiq" }, 403);
      const list = await env.OVOZ_FEED.list({ prefix: "job:", limit: 200 });
      const items = await Promise.all(list.keys.map((k) => env.OVOZ_FEED.get(k.name, "json")));
      return json({ items: items.filter(Boolean).sort((a, b) => (b.at || 0) - (a.at || 0)) });
    }

    if (action === "audio") {
      if (!feedAllowed(req, env)) return new Response("Yopiq", { status: 403 });
      const got = await env.OVOZ_FEED.getWithMetadata(`audio:${id}`, { type: "arrayBuffer" });
      const buf = got?.value;
      if (!buf) return new Response("Topilmadi yoki muddati tugagan", { status: 404 });

      // Eski yozuvlarda metadata yo'q — turni baytlardan aniqlaymiz.
      const type = got.metadata?.type || sniffType(new Uint8Array(buf.slice(0, 16)));
      const total = buf.byteLength;
      // Pleyer faylni bo'lak-bo'lak so'raydi va davomiylikni shu orqali
      // aniqlaydi. Accept-Ranges bo'lmasa, u to'liq uzunlikni ko'rsata
      // olmaydi va orqaga-oldinga surib bo'lmaydi.
      const base = { "content-type": type, "accept-ranges": "bytes", "cache-control": "private, max-age=600" };

      const m = (req.headers.get("range") || "").trim().match(/^bytes=(\d*)-(\d*)$/);
      if (m) {
        let start = m[1] === "" ? null : Number(m[1]);
        let end = m[2] === "" ? null : Number(m[2]);
        if (start === null) { start = Math.max(0, total - (end || 0)); end = total - 1; }
        else if (end === null || end >= total) { end = total - 1; }
        if (start > end || start >= total) {
          return new Response("Range noto'g'ri", { status: 416, headers: { ...base, "content-range": `bytes */${total}` } });
        }
        return new Response(buf.slice(start, end + 1), {
          status: 206,
          headers: { ...base, "content-range": `bytes ${start}-${end}/${total}`, "content-length": String(end - start + 1) },
        });
      }
      return new Response(buf, { headers: { ...base, "content-length": String(total) } });
    }

    if (action === "stats") {
      // Statistikada faqat raqamlar bor — mijoz ma'lumoti yo'q. Shuning
      // uchun u FEED_TOKEN dan tashqari oddiy API kaliti bilan ham ochiladi:
      // kuzatuv rejimidagi no_token_by ni o'qish uchun konsol kaliti shart
      // bo'lmasligi kerak. /api/feed va /api/audio esa FEED_TOKEN da qoladi —
      // ularda real suhbat matni va audiosi bor.
      if (!feedAllowed(req, env) && !checkApi(req, env).ok) {
        return json({ error: "Yopiq" }, 403);
      }
      const q = url.searchParams;
      const from = q.get("from") || "0000-00-00";
      const to = q.get("to") || "9999-99-99";
      const list = await env.OVOZ_FEED.list({ prefix: "stat:", limit: 400 });
      const names = list.keys.map((k) => k.name)
        .filter((n) => { const d = n.slice(5); return d >= from && d <= to; });
      const days = (await Promise.all(names.map((n) => env.OVOZ_FEED.get(n, "json"))))
        .filter(Boolean).sort((a, b) => (a.day < b.day ? -1 : 1));
      return json({ days });
    }

    if (action === "authstate") {
      const a = checkApi(req, env);
      return json({
        tokens_configured: a.gated,
        enforcing: a.enforced,
        this_request_ok: a.ok,
        mode: !a.gated ? "ochiq" : a.enforced ? "yoqilgan" : "kuzatuv",
      });
    }

    if (action === "feedcheck") {
      return json({ enabled: Boolean(env.FEED_TOKEN), ok: feedAllowed(req, env) });
    }

    // ---------- RunPod proksisi ----------
    if (!env.RUNPOD_API_KEY) {
      return json({ error: "RUNPOD_API_KEY o'rnatilmagan. Terminalda: npx wrangler secret put RUNPOD_API_KEY --name ovoz-konsoli" }, 503);
    }

    const auth = checkApi(req, env);
    if (!auth.ok && auth.enforced) {
      return json({ error: "Kalit kerak yoki noto'g'ri. Authorization: Bearer <kalit>" }, 401);
    }
    // Kuzatuv hisoblagichi faqat /run uchun — sabab bumpAuth izohida.
    if (auth.gated && action === "run") {
      ctx.waitUntil(bumpAuth(env, req, auth.ok).catch(() => {}));
    }

    let target = null;
    if (action === "health") target = `${BASE}/health`;
    else if (action === "run") target = `${BASE}/run`;
    else if (action === "status" && /^[a-zA-Z0-9-]{8,80}$/.test(id || "")) target = `${BASE}/status/${id}`;
    if (!target) return json({ error: "Noma'lum yo'l" }, 404);

    let body = null, input = null;
    if (action === "run") {
      body = await req.text();
      try { input = JSON.parse(body).input || null; } catch { /* RunPod o'zi xato qaytaradi */ }
    }

    let res;
    try {
      res = await fetch(target, {
        method: action === "run" ? "POST" : "GET",
        headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, "Content-Type": "application/json" },
        body,
      });
    } catch (e) {
      return json({ error: `RunPod'ga ulanib bo'lmadi: ${e.message}` }, 502);
    }

    const text = await res.text();
    let data = null;
    try { data = JSON.parse(text); } catch { /* JSON emas — o'zgartirmasdan uzatamiz */ }

    // ---------- oqimga yozish ----------
    if (data && env.OVOZ_FEED) {
      if (action === "run" && data.id && input) {
        const audio = await saveAudio(env, ctx, data.id, input);
        await saveEntry(env, data.id, {
          at: Date.now(), status: "IN_QUEUE", audio,
          opts: { beam: input.beam_size, vad: input.vad, lang: input.language, hotwords: Boolean(input.hotwords) },
        });
      } else if (action === "status" && data.status) {
        const out = data.output || {};
        const done = ["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"].includes(data.status);
        const entry = await saveEntry(env, id, {
          status: data.status,
          queue_sec: (data.delayTime || 0) / 1000,
          ...(out.status === "success"
            ? { text: out.text, duration: out.duration_sec, proc: out.processing_time_sec, rtf: out.realtime_factor }
            : out.message ? { error: out.message } : {}),
        });
        if (done && !entry.counted) {
          await bumpStats(env, entry, out, out.status !== "success");
          await saveEntry(env, id, { counted: true });
        }
      }
    }

    return new Response(text, {
      status: res.status,
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  },
};
