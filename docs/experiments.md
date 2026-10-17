# Kísérletnapló – janus_01

Minden futás egy bejegyzés: mi volt a beállítás, mi jött ki, mit tanultunk belőle. A döntések indoklása a `decisions.md`-ben van, itt csak az eredmények.

## Run 1 – baseline, 2026-10-15 (Kaggle, T4)

**Kód:** commit `08b3e03`. **Modell:** JanusCNN (23 585 paraméter). **Beállítás:** 10 epoch, batch 64, Adam lr 1e-3, nincs ütemező, pos_weight 0.113, audio config: sr 16000, max_sec 4.0, n_fft 1024, hop 256, n_mels 80, trim_db 30.

**Adat:** train 25 380 (spoof arány 0.898), dev 24 844 (0.897).

| Epoch | Train loss | Dev EER |
|---|---|---|
| 1 | 0.1199 | 21.60% |
| 2 | 0.0880 | 13.91% |
| 3 | 0.0639 | 8.64% |
| 4 | 0.0501 | 20.60% |
| 5 | 0.0439 | 12.79% |
| 6 | 0.0382 | 6.93% |
| 7 | 0.0362 | 13.25% |
| 8 | 0.0335 | 5.45% |
| 9 | 0.0302 | 6.33% |
| 10 | 0.0294 | **4.88%** |

**Legjobb dev EER: 4.88% (10. epoch).** Checkpoint: `janus_best.pt` (305 KB), a Kaggle-kernel Output fülén.

**Idő:** 1. epoch ~23 perc (cache-építés, CPU-kötött), 2–10. epoch ~16 s/epoch. Teljes tanítás a cache után ~3 perc.

**Tanulságok:**
- A train loss egyenletesen csökken, nincs instabilitás. A BatchNorm + pos_weight kombináció működik.
- Nincs túltanulás: a legjobb EER az utolsó epochban jött, a modell alultanított. Több epoch kell.
- A dev EER erősen ugrál (8.6 → 20.6 → 12.8 → 6.9 → 13.3). Az állandó lr 1e-3 valószínűleg túl nagy a késői epochokra. Ütemezővel várhatóan simább.
- A dev halmaz ugyanazokat a támadásokat (A01–A06) tartalmazza, mint a train. A 4.88% optimista; az eval halmazon (A07–A19, ismeretlen támadások) rosszabb lesz.
- Viszonyítás: az ASVspoof 2019 hivatalos LFCC-GMM baseline dev EER-je 2.71%, eval 8.09%.
- A cache a `/tmp`-ben elveszett a kernel leállásával; minden új futás 23 perccel indul, amíg nincs megőrzött cache.

**Következő lépések:** (1) eval halmaz kiértékelése támadástípusonként ezzel a checkpointtal, (2) 30 epoch + lr-ütemező, (3) cache megőrzése Kaggle-adathalmazként.
