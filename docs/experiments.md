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

## Run 2 – augmentáció + cosine, 2026-10-18 (Kaggle, T4)

**Kód:** commit `9ea8c16`. **Változás Run 1-hez képest:** `--augment` (SpecAugment a train halmazon: zajpadló −50…−20 dB, 2 frekvenciamaszk max 8 sor, 2 időmaszk max 20 oszlop), `--sched cosine`, 30 epoch. Minden más azonos.

**Hipotézis volt:** az eval EER érdemben 24,3% alá megy, az A10/A12/A13/A15/A17 sorok javulnak. A dev EER kicsit romolhat.

**Eredmény: a hipotézis nem jött be. Az augmentáció túl erős volt.**

| Epoch | lr | Train loss | Dev EER |
|---|---|---|---|
| 2 | 1.0e-3 | 0.1375 | 33.51% |
| 5 | 9.6e-4 | 0.1238 | 27.63% |
| 6 | 9.3e-4 | 0.1193 | 47.99% |
| 10 | 7.9e-4 | 0.1029 | 20.95% |
| 15 | 5.5e-4 | 0.0871 | 23.78% |
| 19 | 3.5e-4 | 0.0781 | 18.62% |
| 21 | 2.5e-4 | 0.0725 | 18.33% |
| 24 | 1.3e-4 | 0.0689 | **16.77%** |
| 27 | 4.3e-5 | 0.0668 | 17.40% |
| 30 | 2.7e-6 | 0.0667 | 16.92% |

**Legjobb dev EER: 16.77% (24. epoch).** Run 1-ben 4.88% volt 10 epoch után. A train loss 30 epoch után 0.067, Run 1-ben 10 epoch után 0.029.

**Eval, támadásonként (Run 1 → Run 2):**

| Eval támadás | Run 1 | Run 2 | |
|---|---|---|---|
| A07 | 4.20% | 26.89% | sokkal rosszabb |
| A08 | 1.56% | 12.60% | rosszabb |
| A09 | 2.33% | 7.99% | rosszabb |
| A10 (Tacotron2 + WaveRNN) | 35.91% | 44.75% | rosszabb |
| A11 | 3.82% | 44.42% | sokkal rosszabb |
| A12 (Tacotron + WaveNet) | 56.31% | 39.30% | jobb |
| A13 (VC + TTS, hullámforma-szűrés) | 57.05% | 12.35% | sokkal jobb |
| A14 | 13.90% | 31.02% | rosszabb |
| A15 (VC, WaveNet) | 34.99% | 26.51% | jobb |
| A16 | 6.73% | 31.01% | sokkal rosszabb |
| A17 (VAE VC, hullámforma-szűrés) | 35.01% | 22.08% | jobb |
| A18 | 4.60% | 25.67% | sokkal rosszabb |
| A19 | 4.54% | 22.94% | sokkal rosszabb |

Az összesített eval EER nem került a naplóba; a per-attack sorokból látszik, hogy 24,3%-nál rosszabb.

**Tanulságok:**
- **A zajpadló volt a hiba.** A `mel_db` a fájl maximumához normalizál: a leghangosabb pont 0 dB, a padló −80 dB. A −50…−20 dB-es zaj ebből mindent elfed, ami a zaj szintje alatt van. −20 dB-nél ez a spektrogram nagy része. A modell a tanítás alatt soha nem látott tiszta spektrogramot, a dev és az eval viszont tiszta. Ez tanítás–teszt eltérés, nem regularizáció.
- **A modell "elfelejtette" a finom nyomokat.** Ahol Run 1 jó volt (A07, A08, A11, A16, A18, A19: 2–7%), ott most 8–44%. Ezek a támadások a halk tartományban, a finom spektrális szerkezetben különböznek a valódi hangtól; a zaj pont ezt törölte.
- **A nehéz támadásokon viszont javult** (A12 56 → 39%, A13 57 → 12%, A15 35 → 27%, A17 35 → 22%). A "tiszta = valódi" szabály tényleg megszűnt, ahogy vártuk. Tehát az irány jó, az adag rossz.
- **A dev EER továbbra is ugrál** az első 10 epochban (33 → 36 → 28 → 48 → 27%), az utolsó 10-ben már simább (16,8–19,1%). A cosine ütemező a vége felé azt csinálja, amit kell. Az eleji ugrálás az erős augmentációból jön, nem az lr-ből.
- **Két dolgot változtattunk egyszerre** (augmentáció + ütemező), ezért a két hatás nem válik szét tisztán. Az ütemezőt megtartjuk, az augmentációt gyengítjük; így Run 3 egy változtatás lesz Run 2-höz képest.

**Következő lépés (Run 3):** a zajpadló −70…−45 dB-re (csak a legcsendesebb részeket érinti), a maszkok maradnak. Cosine, 30 epoch marad. Várakozás: dev EER visszamegy 5–8% közé, az eval a könnyű támadásokon Run 1 közelébe, a nehezeken Run 2 közelébe.

## Run 3 – gyengébb zajpadló, 2026-10-18 (Kaggle, T4)

**Kód:** commit `9c129ce`. **Változás Run 2-höz képest:** zajpadló −70…−45 dB (volt −50…−20). Maszkok, cosine, 30 epoch változatlan.

**Hipotézis volt:** dev EER 5–8%, eval a könnyű támadásokon Run 1 közelében, a nehezeken Run 2 közelében.

| Epoch | lr | Train loss | Dev EER |
|---|---|---|---|
| 20 | 3.0e-4 | 0.0584 | 8.71% |
| 21 | 2.5e-4 | 0.0566 | 8.44% |
| 24 | 1.3e-4 | 0.0534 | 7.62% |
| 25 | 9.6e-5 | 0.0523 | **6.99%** |
| 28 | 2.5e-5 | 0.0505 | 7.10% |
| 30 | 2.7e-6 | 0.0500 | 7.06% |

**Legjobb dev EER: 6.99% (25. epoch).** Run 1: 4.88%, Run 2: 16.77%. Az utolsó 10 epoch 7–9% között, nem ugrál; a cosine vége simán konvergál.

**Eval összesítve: 23.52%** (Run 1: 24.30%). Gyakorlatilag ugyanaz.

| Eval támadás | Run 1 | Run 2 | Run 3 | |
|---|---|---|---|---|
| A07 | 4.20% | 26.89% | 19.95% | rosszabb, mint Run 1 |
| A08 | 1.56% | 12.60% | 4.10% | visszajött |
| A09 | 2.33% | 7.99% | 3.75% | visszajött |
| A10 (Tacotron2 + WaveRNN) | 35.91% | 44.75% | 42.49% | rossz maradt |
| A11 (Griffin-Lim TTS) | 3.82% | 44.42% | 43.97% | sokkal rosszabb, mint Run 1 |
| A12 (Tacotron + WaveNet) | 56.31% | 39.30% | 50.35% | véletlen körül |
| A13 (VC + TTS, hullámforma-szűrés) | 57.05% | 12.35% | 22.09% | sokkal jobb, mint Run 1 |
| A14 | 13.90% | 31.02% | 9.37% | jobb |
| A15 (VC, WaveNet) | 34.99% | 26.51% | 32.29% | rossz maradt |
| A16 | 6.73% | 31.01% | 11.87% | kicsit rosszabb |
| A17 (VAE VC, hullámforma-szűrés) | 35.01% | 22.08% | 20.37% | jobb |
| A18 | 4.60% | 25.67% | 7.27% | nagyjából visszajött |
| A19 | 4.54% | 22.94% | 6.69% | nagyjából visszajött |

**Idő:** tanítás 30 epoch 36 perc (cache-építéssel), eval 37 perc. Összesen 74 perc.

**Tanulságok:**
- **A dev-hipotézis bejött** (6.99%), az eval-hipotézis csak félig. A nehéz támadások közül A13 és A17 jobb, mint Run 1-ben, de A11 ugyanúgy 44%, mint Run 2-ben, és A07 is 20%.
- **Az összesített eval nem mozdult** (24.3 → 23.5%). Az augmentáció átrendezte, melyik támadás könnyű és melyik nehéz, de nem csökkentette a hibát. Amit nyertünk A13-on és A17-en, azt elvesztettük A11-en és A07-en.
- **A11 (Griffin-Lim) tanulsága:** ez a támadás a halk zajpadlójáról ismerhető fel. Már a gyenge zaj-augmentáció is pont ezt a nyomot törli, ezért mind Run 2-ben, mind Run 3-ban 44%. A zajpadló-augmentáció tehát nem semleges: bizonyos támadások nyomát szisztematikusan eltünteti.
- **A dev és az eval között továbbra is háromszoros a különbség** (7% vs 23,5%). Ez a fő probléma, és a spektrogram-augmentáció önmagában nem oldja meg. A 23 ezer paraméteres modell 80 mel-sávon nem lát elég részletet ahhoz, hogy általános spoof-jellemzőket tanuljon; a mel-skála pont a magas frekvenciákat tömöríti össze, ahol a vocoder-nyomok vannak.
- **Három futás, három rendszer, azonos ~24%.** Ha a következő változtatás sem visz 20% alá, nem a tanítási recepten kell tovább csavarni, hanem a bemeneten vagy a modell méretén.

**Következő lépés (Run 4):** a bemenetet változtatjuk, nem a tanítást. Mel helyett lineáris frekvenciaskálájú spektrogram (vagy LFCC), hogy a magas frekvenciás vocoder-nyomok ne tömörödjenek össze. Az ASVspoof 2019 irodalomban az LFCC-alapú rendszerek következetesen jobbak a mel-alapúaknál ezen a halmazon. Augmentáció: a Run 3 beállítás marad.

## Run 4 – LFCC bemenet, 2026-10-18 (tervezett)

**Változás Run 3-hoz képest:** `config.yaml` `audio.feature: lfcc` (20 LFCC + Δ + ΔΔ = 60 sor, 10 ms lépés, 401 oszlop). Augmentáció: csak maszkolás (a zajpadló LFCC-nél kimarad). Modell, cosine, 30 epoch, batch, lr változatlan. Commit: (push után).

**Hipotézis:** eval EER 20% alá. A11 visszamegy 10% alá (a zaj-augmentáció megszűnt, és a lineáris skála jobban mutatja a Griffin-Lim nyomait). A dev EER 5% körül.

**Eredmény:** (futás után)
