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

## Run 1 – kiértékelés, 2026-10-17 (Kaggle, T4)

**Checkpoint:** Run 1, 10. epoch. **Script:** `src/evaluate.py`, commit `cbad3f4`.

**Dev: 4.88%** (egyezik a tanításkor mérttel, a checkpoint és a feature-számítás konzisztens).

| Dev támadás | EER |
|---|---|
| A01 (WaveNet TTS) | 1.64% |
| A02 (WORLD TTS) | 2.46% |
| A03 (WORLD TTS) | 2.55% |
| A04 (konkatenatív TTS) | 9.77% |
| A05 (WORLD VC) | 4.58% |
| A06 (spektrális szűrés VC) | 3.80% |

**Eval: 24.30%** (7 355 bonafide, 63 882 spoof). Viszonyítás: LFCC-GMM baseline eval 8.09%.

| Eval támadás | EER | |
|---|---|---|
| A07 | 4.20% | jó |
| A08 | 1.56% | jó |
| A09 | 2.33% | jó |
| A10 (Tacotron2 + WaveRNN) | 35.91% | rossz |
| A11 | 3.82% | jó |
| A12 (Tacotron + WaveNet) | 56.31% | véletlennél rosszabb |
| A13 (VC + TTS, hullámforma-szűrés) | 57.05% | véletlennél rosszabb |
| A14 | 13.90% | közepes |
| A15 (VC, WaveNet) | 34.99% | rossz |
| A16 | 6.73% | jó |
| A17 (VAE VC, hullámforma-szűrés) | 35.01% | rossz |
| A18 | 4.60% | jó |
| A19 | 4.54% | jó |

**Tanulságok:**
- Dev és eval között ötszörös különbség: a modell a train halmaz hat rendszerének nyomait tanulta meg, nem általános spoof-jellemzőket. Ahol az eval támadás hasonló technológiájú, ott jó; ahol új neurális vocoder (WaveNet, WaveRNN) készítette, ott nem.
- A12 és A13 56–57%: ezeket a hamisítványokat a modell átlagosan valódibbnak pontozza, mint az igazi felvételeket. Nem "nem tudja", hanem rossz irányba biztos.
- A04 a dev halmazon is a legnehezebb (9.77%): konkatenatív szintézis, valódi hullámforma-darabokból, ezért a legkevesebb vocoder-nyom.
- A17 az irodalomban is a legnehezebb támadás; ott a legjobb rendszerek is 10–30% körül vannak.
- Ez a várt viselkedés egy kis, augmentáció nélküli baseline-nál. Most van viszonyítási alap a generalizációt javító lépésekhez.

**Hiba, javítva:** a notebook rekurzív glob-ja a teljes adathalmazt bejárta a mappák megkereséséhez: 80 perc a DATA_ROOT, további 54 perc a checkpoint. 2 óra 15 perc felesleges várakozás. Commit `0c7ccd4`: szélességi keresés max 5 szint mélyen, nem jár be fájlokat.

**Idő (javítás után várható):** dev cache ~15 perc, eval cache ~45 perc, pontozás pár perc.

**Következő lépések, sorrendben:**
1. Generalizáció javítása: adat-augmentáció a tanításban (véletlen erősítés, zaj, időbeli eltolás; később RawBoost-szerű). Cél: a modell ne a konkrét vocoder-nyomokra tanuljon rá.
2. 30 epoch + lr-ütemező (a dev-görbe ugrálása miatt).
3. Mérés mindig az eval halmazon, támadásonként. A dev EER önmagában félrevezető.

## Run 2 – augmentáció + cosine, 2026-10-18 (tervezett)

**Változás Run 1-hez képest:** `--augment` (SpecAugment a train halmazon, config.yaml `augment` szekció), `--sched cosine`, 30 epoch. Minden más azonos.

**Hipotézis:** az eval EER érdemben 24,3% alá megy, az A10/A12/A13/A15/A17 sorok javulnak. A dev EER kicsit romolhat.

**Eredmény:** (futás után)
