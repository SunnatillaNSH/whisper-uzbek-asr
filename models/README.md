# Pod'dan ko'chirilgan modellar

Bu katalog `.gitignore` da — modellar gigabaytlab va repo ochiq.

## Round 5 da nima ko'chirildi va nega

Ko'chirilgani: **CT2 model** (ishlatish uchun kerak bo'lgani) va **LoRA
adapteri** (~100 MB). Birlashtirilgan (merged) fp16 model ko'chirilmadi —
u ~3 GB va adapterdan qayta yig'iladi, ya'ni hech narsa yo'qolmaydi.
Pod soatiga pul turgani uchun ortiqcha 3 GB ni ko'chirish byudjetdan
yeyishi mumkin edi.

## Merged fp16 ni qayta yig'ish

Kerak bo'lgan hammasi:

* baza model: `Sunnat0091/whisper-large-v3-uz` (round-2) — **OCHIQ repo**,
  token kerak emas
* adapter: `models/round5/adapter/` (shu yerda)
* `peft` 0.21.0, `transformers` 4.57.6 (Pod'dagi versiyalar; `<5` bo'lishi
  shart — 5.x saqlagan tokenizer'ni 4.x o'qiy olmaydi)

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

m = WhisperForConditionalGeneration.from_pretrained("Sunnat0091/whisper-large-v3-uz")
m = PeftModel.from_pretrained(m, "models/round5/adapter")
m = m.merge_and_unload().half()
m.save_pretrained("models/round5/merged")
```

Tokenizer'ni baza modeldan oling (`openai/whisper-large-v3`), round-2
repo'sidagisini EMAS: u transformers 5.x formatida saqlangan va 4.x uni
o'qiy olmaydi (`'list' object has no attribute 'keys'`). LoRA faqat
`q_proj`/`v_proj` ga tegadi, ya'ni tokenizer baza bilan aynan bir xil.

## CT2 ga o'girish

Production bilan bir xil bo'lishi uchun aynan shu buyruq:

```bash
ct2-transformers-converter --model models/round5/merged \
  --output_dir models/round5/ct2 --quantization float16 \
  --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
               vocab.json merges.txt special_tokens_map.json \
               added_tokens.json normalizer.json
```
