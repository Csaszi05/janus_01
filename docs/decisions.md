# Döntésnapló – janus_01

Folyamatos napló arról, hogy mit miért csináltunk így. Új bejegyzés mindig a végére kerül.

## 2026-10-15 – Környezet

**Tanítás Kaggle-ön, nem lokálisan.** A LA adathalmaz (~25 ezer train fájl) és a GPU-igény miatt. Ebből következik:
- Minden adatútvonal paraméter (`--data_root`, `config.yaml`), semmi nincs beégetve.
- Lokálisan nincs adat (`data/LA` hiányzik), itt csak a kódot szerkesztjük és szintetikus hanggal füstpróbázunk.
- A feature-cache a Kaggle munkakönyvtárába kerül (`out/cache`), nem feltételezünk tartós lokális tárolót.

## 2026-10-15 – Csendvágás (`trim_db`)

**Döntés:** a `load_audio` a 4 másodperces fix hosszra vágás ELŐTT levágja a csendet a felvétel elejéről és végéről (`librosa.effects.trim`, `top_db=30`). Az érték a `config.yaml`-ban van (`audio.trim_db`).

**Miért:** az ASVspoof fájlok elején gyakran fél-egy másodperc csend van. Fix 4 mp-es kivágásnál ez elveszi a beszéd helyét. Vágás után a 4 mp tényleg hasznos jelet tartalmaz.

**Miért 30 dB:** librosa alapérték, beszédnél bevett kiindulópont. Nagyobb érték kevesebbet vág, kisebb agresszívabb.

**Ismert mellékhatás:** egyes spoof-rendszerek a csendet is gyanúsan generálják, ezért a csend hossza/jellege önmagában árulkodó lehet. Vágással ezt az információt eldobjuk. A baseline-hoz elfogadható; később érdemes összevetni vágással és anélkül.

**Elvetett alternatíva:** csak az elejéről vágni. Egyszerűbb lett volna, de a librosa trim egyben kezeli mindkét véget, és a végén lévő csend is ugyanúgy pazarol helyet.

## 2026-10-15 – Modell (`src/model.py`, `JanusCNN`)

**Döntés:** minimális 2D CNN, három azonos blokk (Conv3×3 → BatchNorm → ReLU → MaxPool2), csatornák 1→16→32→64, majd AdaptiveAvgPool → Dropout(0.3) → Linear(64→1). Kimenet egyetlen logit mintánként.

**Miért ilyen kicsi:** ez baseline. A cél egy gyorsan tanítható, értelmes számot adó referencia, amihez képest mérni lehet a későbbi javításokat. Kaggle GPU-időkeretbe is beleférjen.

**Miért 2D CNN mel-spektrogramon:** a spektrogram képként kezelhető, a spoof-artefaktumok (pl. vocoder nyomok, frekvenciasáv-hiányok) lokális mintázatok, amit a konvolúció jól fog.

**AdaptiveAvgPool miért:** így a modell tetszőleges hosszú bemenettel működik, nem csak pontosan 4 mp-essel. Később, ha változtatunk a `max_sec`-en, nem kell a modellhez nyúlni.

**Egy logit, nem két osztály softmax:** bináris feladatnál ugyanaz, de egyszerűbb, és a `BCEWithLogitsLoss` `pos_weight`-je csak így használható.

**Nincs sigmoid a modellben:** a loss függvény (`BCEWithLogitsLoss`) belül számolja, numerikusan stabilabban. Következtetéskor, ha valószínűség kell, `torch.sigmoid(logit)`.

**Elvetett alternatívák:** RawNet2 / AASIST (nyers hullámformán dolgozó, erősebb, de bonyolultabb modellek) – ezek lehetnek a következő lépés, miután a baseline számot látjuk.

## 2026-10-15 – Adatbetöltő (`MelDataset`)

**Döntés:** a `MelDataset` a meglévő `load_protocol` DataFrame-jéből dolgozik; a `load_protocol` érintetlen maradt. Csak a szükséges oszlopokat (path, file, y) tartja meg listaként.

**Miért a DataFrame-ből és nem újraolvasva:** egy helyen van a protokoll-logika; a notebook és a tanítás ugyanazt az adatot látja.

**Cache:** ha `cache_dir` van megadva, a mel-spektrogram fájlnevenként `.npy`-ba kerül. Első epoch kiszámolja, a továbbiak csak betöltik. A mel-számítás (librosa, CPU) a leglassabb lépés, GPU-n tanítva ez lenne a szűk keresztmetszet.

**Cache kulcs csak a fájlnév.** Ha az `audio` config változik (sr, n_mels, trim_db stb.), a cache-t kézzel törölni kell (`out/cache`), különben régi spektrogramokat tölt be. Tudatos egyszerűsítés; ha zavaró lesz, a config hash-ét bele lehet tenni a könyvtárnévbe.

**Float32 mindenhol:** a spektrogram és a címke is float32, mert a `BCEWithLogitsLoss` float címkét vár, és a GPU-n a float32 az alapértelmezett.

## 2026-10-15 – Tanító script (`src/train.py`)

**Hely:** `src/`-ben van, nem a gyökérben, hogy a `from dataset import ...` importok ugyanúgy működjenek, mint a notebookban (Python a script könyvtárát automatikusan a path-ra teszi). Futtatás: `python src/train.py --data_root ...`.

**Osztályegyensúly – `pos_weight`:** a train halmazban kb. 9 spoof jut 1 bonafide-ra. Súlyozás nélkül a modell megtanulhatná, hogy mindent spoof-nak mond. `pos_weight = n_bonafide / n_spoof ≈ 0,11`, azaz a (többségi) spoof osztály veszteségét lecsökkentjük. A számot az adatból számoljuk, nem beégetett.

**Adam, lr 1e-3:** szabványos kiindulópont, kis CNN-hez általában jó. Nincs ütemező (scheduler) – baseline-ban fölösleges bonyolítás.

**EER számítása:** `sklearn.metrics.roc_curve` adja az FPR és TPR görbét; FNR = 1 − TPR; az EER ott van, ahol |FNR − FPR| minimális, és a kettő átlagát vesszük. Ez a bevett közelítés (a görbe diszkrét, a pontos metszéspont nem feltétlenül mintapont).

**Checkpoint csak javuláskor:** a legjobb dev-EER-nél mentünk (`out/janus_best.pt`), benne modell, optimizer, epoch, config. Így a tanítás végén a legjobb állapot van meg, nem az utolsó (ami túltanulhatott). A config is benne van, hogy a betöltésnél tudjuk, milyen audio-beállítással készült.

**Dev halmazon mérünk, nem eval-on:** az eval halmaz 13 olyan támadást tartalmaz, ami a train-ben nincs. Ezt a végső teszthez tartjuk fenn; ha epochonként ezt néznénk, implicit ráhangolnánk.

**Log formátum:** soronként egy epoch, `*` jelöli a checkpoint-mentést. Kaggle-ön a kimenet a notebook cellában látszik, bonyolultabb loggolás (TensorBoard, W&B) egyelőre nem kell.

**`num_workers=2`:** a Kaggle-kernelnek jellemzően 2-4 CPU-ja van. Paraméter (`--workers`), ha más környezetben futtatjuk.

**Nincs seed rögzítés, nincs early stopping, nincs mixed precision.** Tudatosan kihagyva; akkor jön, ha a baseline számot már látjuk és reprodukálhatóságra / gyorsításra van szükség.

## 2026-10-15 – Egyéb

- `requirements.txt`: hozzáadva `scikit-learn` (EER), `numpy` (explicit, bár a többi is behúzza).
- `.gitignore` már tartalmazza a `*.npy` és `*.pt` mintákat, a cache és a checkpoint nem kerül gitbe.
- Füstpróba lokálisan szintetikus wav fájlokkal: betöltés + trim + mel + cache + modell + EER összeáll. Valódi adaton csak Kaggle-ön fut.

## 2026-10-15 – Modell-szimuláció (`docs/janus_sim.html`)

**Mi ez:** önálló HTML oldal, ami a `JanusCNN` pontos mását futtatja a böngészőben (ugyanazok a rétegek, méretek, 23 585 paraméter), véletlen súlyokkal, szintetikus spektrogramon. Tanulási segédanyag, nem része a tanítási kódnak. Megnyitás: dupla kattintás a fájlra, nem kell szerver.

**Miért böngészőben, nem notebookban:** interaktív (lépésenkénti konvolúció, csatornák kattintása, súlyok újrasorsolása), és nem kell hozzá Python környezet vagy adat. Kaggle nélkül is nézhető.

**Korlátok, tudatosan:** a bemenet szintetikus, a spoof-nyomok eltúlzottak; a BatchNorm egyetlen mintán számol, nem kötegen; a súlyok tanítatlanok, a kimenet nem jelent semmit. Mindez az oldalon is fel van tüntetve.

## 2026-10-15 – Kaggle-futtatás (`notebooks/02_train_kaggle.ipynb`)

**Kód Kaggle-re: `git clone` GitHubról.** A kernel klónozza a repót, nem másoljuk be a kódot cellákba. Egy forrás van, nem kell szinkronban tartani. Ára: a kernelnek internet kell, és minden lokális változást előbb pusholni kell.

**Mel-cache a `/tmp`-be, nem a `/kaggle/working`-be.** Train + dev kb. 50 ezer spektrogram, ~4 GB. A `/kaggle/working` a futás végén kimenetként mentődik, ezt fölösleges 4 GB-tal terhelni. A `/tmp` a kernel leállásával elvész, de a következő futás újraszámolja (az első epoch ára). Ezért lett a `--cache_dir` külön paraméter.

**Köztes loss-kiírás (`--log_every 100`).** Az első epoch a cache-építés miatt 10–20 perc lehet, kimenet nélkül nem látszik, hogy él-e. 100 batch-enként egy sor.

**10 epoch, batch 64, lr 1e-3.** Kiinduló értékek a spec alapján. Ha a dev EER az 5. epoch után még javul, több epoch; ha már a 2–3. után romlik, túltanulás, kevesebb epoch vagy erősebb regularizáció.

**A checkpoint `weights_only=False`-szal töltődik.** A checkpointban a config dict is benne van, az újabb torch alapból csak tenzorokat enged betölteni. Saját fájlunk, biztonságos.

## 2026-10-17 – Kiértékelés (`src/evaluate.py`, `notebooks/03_eval_kaggle.ipynb`)

**Külön script, nem a train.py része.** A kiértékelés más életciklusú: egy checkpointot többször, több halmazon is futtatunk, és az eval halmaz 71 ezer fájlja miatt önmagában is hosszú. Így a tanítást nem terheli.

**Az audio config a checkpointból jön, nem a config.yaml-ból.** A spektrogramot pontosan úgy kell számolni, ahogy a modell tanulta. Ha közben átírjuk a config.yaml-t, a kiértékelés akkor is a tanításkori beállítást használja. Ezért került bele a config a checkpointba.

**Támadásonkénti EER: bonafide vs. egy-egy támadás.** Minden A07–A19 rendszerre külön EER, mindig a teljes bonafide halmazzal szemben. Ez az ASVspoof-irodalom szokásos bontása, és megmutatja, melyik támadástípus fog ki a modellen.

**Pontszámok CSV-be.** Minden fájl logitja mentve (`eval_scores.csv`). Későbbi elemzés (küszöbválasztás, hibaelemzés, min t-DCF) ebből megy, nem kell újra pontozni.

**min t-DCF egyelőre nincs.** Az ASVspoof másik hivatalos metrikája, de hozzá az ASV-rendszer pontszámai kellenek, amik külön fájlban vannak. Az EER-bontás most elég a modell fejlesztéséhez; a t-DCF a végső összehasonlításhoz jön.

**A notebook először a dev halmazon fut.** A tanításkor mért 4.88%-ot kell visszaadnia. Ez ellenőrzi, hogy a checkpoint és a feature-számítás konzisztens, mielőtt egy órát töltünk az eval halmazzal.

## 2026-10-18 – Run 2: augmentáció és ütemező

**Miért augmentáció:** Run 1 eval eredménye (24,3%, A12/A13 56–57%) azt mutatta, hogy a modell a train halmaz hat rendszerének vocoder-nyomait tanulta meg, nem általános spoof-jellemzőket. A "tiszta = valódi" szabály a modern neurális vocodereken fordítva sül el. Az augmentáció ezt a szabályt teszi használhatatlanná, és mélyebb jellemzők tanulására kényszerít. A szakirodalomban ez a legolcsóbb, legjobban bevált első lépés.

**A spektrogramon, nem a hangon.** A cache a tiszta mel-spektrogramot tárolja; az augmentáció betöltéskor, memóriában történik. Így a cache érvényes marad, és minden epochban más torzítást kap a minta. Hangon augmentálni minden epochban 23 perces újraszámolást jelentene.

**Három művelet (SpecAugment-szerű):** zajpadló hozzáadása teljesítmény-tartományban (−50…−20 dB, a fájl maximumához képest), 2 frekvenciasáv-maszk (max 8 mel-sor), 2 időmaszk (max 20 oszlop). A maszk kitöltő értéke a spektrogram átlaga, nem −80: a nulla-energiás lyuk önmagában tanulható nyom lenne.

**Nincs hangerő-augmentáció.** A `mel_db` a fájl maximumához normalizál (`ref=np.max`), ezért a hangerő eleve kiesik. Felesleges lenne.

**torch véletlen, nem numpy.** A DataLoader worker-ek numpy-seedje azonos lenne (minden worker ugyanazt a torzítást adná), a torch worker-enként külön seedet kap.

**Csak a train halmazon.** Dev és eval tiszta marad; a mérésnek a valódi adatot kell mutatnia.

**Cosine ütemező, 30 epoch.** Run 1-ben a dev EER epochról epochra ugrált (8,6 → 20,6 → 12,8) állandó lr mellett. A cosine simán nullára viszi az lr-t, az utolsó epochok finomhangolnak. Nincs paramétere, nem kell hangolni. Alternatíva lett volna a ReduceLROnPlateau, de az a zajos dev EER-re reagálna.

**Kapcsolók, nem alapértelmezés.** `--augment` és `--sched cosine` külön kapcsolók, Run 1 változatlan parancssorral reprodukálható.

**Checkpoint-választás továbbra is dev EER alapján.** Az eval halmaz tanítás közben tiltott; ha azon választanánk, az eredmény nem lenne hiteles. Számítunk rá, hogy a dev EER kicsit romlik az augmentáció miatt, ez elfogadható.
