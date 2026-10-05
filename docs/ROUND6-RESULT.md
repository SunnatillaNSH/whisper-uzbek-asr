# Round 6 — natija (2026-10-05): SALBIY, joylashtirilmasin

**Xulosa.** Faqat qo'ng'iroq (2296 namuna, 3000 qadam, L40S) eval-120 da production'dan **+5.84 punkt YOMON** (36.39% → 42.23%), ishonchli (p=0.000). Production almashtirilmaydi. Round 5 (+8.17) dan kam yomon, lekin yo'nalish bir xil.

## Dataset (`data/round6-dataset/`, gitignore'da)
- v2 (Jev A-004: F1 dev MIJOZ bo'yicha, F2 Claude-tahrirlangan namunalar olib tashlandi). Manba: SellUp D1 Muxlisa transkriptlari + audio, ≤30 s, ≥1.2 s, cps 6–26.
- train **2296** namuna (14.57 soat, 438 qo'ng'iroq) / dev **266** (1.72 soat, 50 qo'ng'iroq, 21 mijoz). Tashqi korpus YO'Q (UzbekVoice = 0), tiklash bosqichi yo'q.
- Assertlar (train ∩ dev ∩ eval-120 = 0) Pod'da `train_calls.py` boshida o'tdi; `round6_send.sh` hajm tekshiruvi 2296/266/310 o'tdi.

## Sozlamalar va to'xtash
Boshlang'ich: `Sunnat0091/whisper-large-v3-uz` (round-2), LoRA r (15.7M parametr, 1.01%), batch 8, LR 5e-5 chiziqli, warmup 100, MAX_STEPS 3000 (≈10.4 epoxa), eval/save har 250, EarlyStopping patience 3, load_best_model_at_end (dev loss), qo'riqchi $3.0.
**To'xtash sababi: MAX_STEPS=3000 ga yetdi** (erta to'xtash ishlamadi). Eng yaxshi checkpoint = 3000 (dev loss 0.6144). 3095 s, 1.03 qadam/s.

Dev loss (qadam: loss): 250: 0.8425, 500: 0.7191, 750: 0.6796, 1000: 0.6569, 1250: 0.6404, 1500: 0.6331, 1750: 0.6265, 2000: 0.6174, 2250: 0.6155, 2500: 0.6145, 2750: 0.6156, 3000: 0.6144. Train loss 3000-qadamda 0.448 (0.741 @500 → 0.448): train-dev farqi o'sdi.

**Uzaytirish qarori (6000 gacha): QILINMADI.** Qoida: dev loss hali tushayotgan va oxirgi 2 eval ketma-ket yaxshilangan bo'lishi kerak. Haqiqatda 2250→2500: −0.0010, 2500→2750: **+0.0011 (yomonlashdi)**, 2750→3000: −0.0012. Ketma-ket yaxshilanish yo'q, LR 3000 da nolga tushadi, 2000 dan keyin jami −0.003. Plato. Qo'shimcha: eval-120 WER production'dan yomon, ya'ni dev loss pasayishi WER ga o'tmagan.

## Natija (eval-120: 310 namuna, 14 015 so'z; production = `analysis/eval120_prod.json`, bir xil `eval_pod.py` dekodlash, handler.py bilan mos)

| | Production | Round 6 | Farq (95% oraliq, p) |
|---|---:|---:|---|
| **eval-120 WER** | 36.39% | **42.23%** | **+5.84** [+4.62, +7.08], p=0.000 |
| eval-120 imlo-norm. WER | 30.23% | 35.06% | +4.83 [+3.60, +6.09], p=0.000 |
| **eval-50 WER** (129 namuna, 6049 so'z; eval-120 ichidan) | 37.84% | **42.25%** | **+4.41** [+2.35, +6.47], p=0.000 |
| eval-50 imlo-norm. | 31.44% | 35.43% | +3.98 [+1.83, +6.15], p=0.0004 |

Juftlashgan bootstrap `scripts/compare_runs.py` (10 000 raund; eval-50 — shu skriptning `score/bootstrap` funksiyalari bilan eval-120 natijalaridan eval50.csv yo'llari bo'yicha). Namuna darajasida: yaxshilandi 78, yomonlashdi 207, teng 25.

**S/D/I (eval-120):** production 2965 / 1333 / 802 (58% / 26% / 15%); Round 6 **3752 / 1054 / 1113** (63% / 17% / 18%). Tushish (D) −279 yaxshi; almashtirish +787 va qo'shish +311 yomon.
eval-50: S 1342→1570, D 575→479, I 372→507.

## Dev loss va WER nomuvofiqligi
Dev loss 0.842 → 0.614 (−27%) bir tekis tushdi, lekin eval-120 WER production'dan 5.84 punkt yomon. Demak dev loss (Muxlisa-yorliqli, mijoz bo'yicha ajratilgan dev bo'yicha token-darajasidagi o'rtacha xato) bu yerda real WER ni bashorat qilmadi: model dev'dagi yorliq uslubini yaxshi bashorat qiladi, lekin beam-5 + VAD bilan to'liq dekodlashda ortiqcha so'z (I +311) va almashtirish (S +787) chiqaradi. Checkpoint tanlash dev loss bo'yicha qilingani uchun (eval to'plamiga qarab tanlanmagan) bu qoida to'g'ri, lekin dev loss ishonchli tanlov mezoni emasligi ko'rindi. Round 6 va production'ni DEV da WER bilan solishtirish nomuvofiqlik sababini ajratadi (pastga qarang).

## Qo'shishlar (I) nega o'sdi — o'lchangan faktlar va farazlar
O'lchangan (eval-120, production → Round 6):
- Gipoteza/yorliq so'z nisbati 0.962 → 1.004: model kam tushirib, ko'proq so'z chiqaradi (D pasaydi, S va I oshdi). Bo'sh gipoteza: 0 / 0.
- Qo'shish bir-ikki namunada to'planmagan: o'sishning faqat 22% i 10 ta namunada, 44% i 30 tada; ≥10 so'z o'sgan namuna 5 ta. Ya'ni bu bir nechta "hallucination" emas, keng tarqalgan.
- Davomiylik bo'yicha: 22–31 s namunalar (211 ta, ko'pchilik) qo'shish 590 → 853, WER 35.3 → 41.3; 15–22 s: 164 → 208, 40.9 → 46.3; 8–15 s: 34 → 41; <8 s (12 ta): 14 → 11, WER 57.6 → 64.6 (kam namuna). Yomonlashuv hamma bucket'da.
- Chetki (birinchi/oxirgi 2 so'z) sof qo'shishlar 235 → 287, o'rtadagilari 129 → 227: o'sishning ~65% i matn o'rtasida.
- 4+ ketma-ket takrorlangan so'zli gipotezalar: 11 → 21 (takrorlanish kuchaydi, lekin bu ham asosiy sabab emas).
Dekodlash shartnomasi sabab EMAS: `eval_pod.py --check-only` Pod'da o'tdi (beam 5, VAD 2000/400, hotwords bo'sh, temperatura zaxirasi, condition_on_previous_text=False — handler.py bilan mos), CT2 production bilan aynan bir xil buyruq (float16, bir xil tokenizer fayllari), ikkala model bir xil skript va bir xil eval.csv da o'lchangan.

Farazlar (TEKSHIRILMAGAN, aniq sabab o'lchanmagan):
1. **Yorliq-audio nomuvofiqligi.** Dataset 55 s li Muxlisa bo'laklaridan >30 s bo'laklarni matn↔vaqt xaritasi bo'yicha kesib tuzilgan (47 s eksport + kesish: forced-alignment YO'Q, `ROUND6-PLAN.md`). Kesish chegarasi bir necha so'zga siljigan bo'lsa, model chegarada ortiqcha/yetishmagan so'zlarni yodlaydi. Biroq qo'shishlarning ko'pi o'rtada — bu faraz hammasini tushuntirmaydi.
2. **Muxlisa yorlig'ining shovqini.** Yorliq mustaqil tizim chiqishi (mutlaq haqiqat emas); 10 epoxa 2296 namunada shu shovqinni yodlaydi (train loss 0.45 vs dev 0.61, dev 2000 dan keyin plato). Production (Round 3/4) 941–1258 namunada, kamroq epoxada to'xtagan. Ortiqcha moslanish (overfitting) faraziy; dev loss yomonlashmagan (U shakli yo'q), shuning uchun bu ham tasdiqlanmagan.
3. Tarkib: dev chiquvchiga og'gan (220/46), train chiquvchi 1678 / kiruvchi 618; eval-120 proporsional. Ta'siri o'lchanmagan.
Foydali keyingi o'lchov (arzon): Round 6 va production'ni DEV (266) da solishtirish — Round 6 o'z dev'ida yaxshi, eval-120 da yomon bo'lsa, muammo yorliq/taqsimot siljishida; ikkalasida ham yomon bo'lsa — o'qitishda. Pod yoki GPU kerak (CT2 Mac'da CPU da sekin).

## Narx
Pod L40S ($1.11/soat, 120 GB container disk, volume yo'q), Pod boshlanish epoch 1791202647. Yuklash ~22.5 daq (843 MB, ~0.6 MB/s), trening 51.6 daq, CT2+eval-120 ~20 daq, Mac'ga nusxalash (adapter + ct2) kechikdi. **Jami Pod: taxminan $2.3** (PM bahosi: epoch 1791202647 dan o'chirilishigacha ~2.1 soat; Pod o'chirilgan vaqtning aniq epoch'i yozilmadi. Qo'riqchi bashorati trening bosqichida $1.99, chegara $3.0, egasi chegarasi $3.5/$4.5 dan oshmadi). Eslatma: to'liq nusxalash (ct2 2.9 GB) ~0.7 MB/s tezlikda ~1 soat olardi, shu sabab to'xtatildi. Endpointga so'rov yo'q, production o'zgarmagan.

## Model joyi
`models/round6/` (gitignore'da, `git check-ignore -q` exit 0): `adapter/` (60 MB), `best.txt`, `eval120_r6.json`, `trainer_state.json`, `pipeline.log`, `SHA256SUMS` (`cd models/round6 && shasum -a 256 -c SHA256SUMS`, 6 fayl OK). **CT2 nusxalanMAGAN**: PM Pod'ni o'chirdi, 2.9 GB ning 2.07 GB i kelgan edi, qisman nusxa o'chirildi. Pod o'chirildi (PM tomonidan brauzer orqali, RunPod ro'yxati bo'sh). CT2 kerak bo'lsa adapter + baza modeldan `models/README.md` retsepti bilan qayta yig'iladi. Per-namuna eval natijalari (matn bor): `analysis/eval120_r6.json` (gitignore'da). Merged fp16 ko'chirilmadi (adapterdan qayta yig'iladi, `models/README.md`).

## Tavsiya
1. **Joylashtirmang**; production Round 3/4 modelida qoladi.
2. Qo'ng'iroq-faqat yo'nalishi ham yaxshilamadi (Round 5 +8.17, Round 6 +5.84): ko'proq Muxlisa-yorliqli namuna (2296 vs 941–1258) WER ni tushirmadi. Navbatdagi qadam — yana trening emas, avval yorliq sifati va chegara nomuvofiqligini tekshirish (yuqoridagi dev solishtiruvi; kesish chegaralarini audio bilan tasdiqlash; forced-alignment).
3. Round 6 ni ensemble/checkpoint tanlash bilan qutqarishga urinmang: eval-120 ga qarab tanlash taqiqlangan, dev loss esa WER ni bashorat qilmadi (0.614 dev loss — WER yomon).
