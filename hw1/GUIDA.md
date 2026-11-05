# HW1 – Guida completa (Sparse GEMM su Raspberry Pi 4)

Consegna ufficiale: https://valentinopeluso.github.io/ECAI27/homework/homework-1.html
**Scadenza: 2 dicembre 2026, ore 23:59**

- [Parte 1: esecuzione rapida sul Raspberry](#parte-1--esecuzione-rapida-sul-raspberry): i comandi da lanciare.
- [Parte 2: capire il progetto](#parte-2--capire-il-progetto): cosa fa il codice e perché.
- [Parte 3: problemi comuni](#parte-3--problemi-comuni).
- [Parte 4: checklist di consegna](#parte-4--checklist-di-consegna).

---

## Parte 1 – Esecuzione rapida sul Raspberry

### 1.1 Portare il codice sul Pi

**Opzione A, con git (consigliata).** Sul Mac fai commit e push, poi sul Pi:

```bash
git clone <url-del-repo> efficient-computing-ai     # la prima volta
# oppure, se il repo è già sul Pi:
cd efficient-computing-ai && git pull
cd hw1
```

**Opzione B, con scp dal Mac.** Sostituisci `pi@raspberrypi.local` con utente e indirizzo del tuo Pi:

```bash
scp -r "/Users/gianmariadifronzo/Desktop/Efficient Computing/efficient-computing-ai/hw1" pi@raspberrypi.local:~/
ssh pi@raspberrypi.local
cd ~/hw1
```

### 1.2 Controlli preliminari (una volta sola)

```bash
gcc -dumpmachine          # DEVE contenere "aarch64" (es. aarch64-linux-gnu)
sudo apt update
sudo apt install -y build-essential python3-matplotlib zip
```

Se `gcc -dumpmachine` restituisce `arm-linux-gnueabihf`, il sistema operativo è a 32 bit e il Makefile si rifiuta di compilare. Serve Raspberry Pi OS **64-bit**.

### 1.3 Un solo comando per tutto

```bash
chmod +x run_pi.sh
GROUP=7 ./run_pi.sh             # sostituisci 7 con il vostro numero di gruppo
# per una deviazione standard più affidabile:
GROUP=7 RUNS=5 ./run_pi.sh
```

Lo script esegue in ordine:

1. `make clean` e compilazione di benchmark e test.
2. `./src/sparse_gemm/test_kernel`: tutte le righe devono essere `PASS`. Se ce n'è una `FAIL`, lo script si ferma.
3. `RUNS` esecuzioni del benchmark, salvate in `results_run1.csv`, `results_run2.csv`, …
4. La creazione di `HW1_Group7.tex` dal template, con il numero del gruppo. Se il file esiste già, ad esempio con i nomi già inseriti, non viene sovrascritto.
5. `analysis/analyze.py`, che genera `analysis/summary.md`, `summary.csv` e 3 grafici `.png`, e **scrive i numeri direttamente nella tabella di `HW1_Group7.tex`**. Poi compila il PDF, se sul Pi c'è `pdflatex`.
6. `HW1_Group7.zip` con tutti i file da consegnare.

Alla fine sul terminale compare la tabella riassuntiva, la stessa di `analysis/summary.md`.

### 1.4 Esecuzione manuale (alternativa allo script)

Sono gli stessi comandi indicati nella consegna:

```bash
make clean
make sparse_gemm
make src/sparse_gemm/test_kernel
./src/sparse_gemm/test_kernel
./src/sparse_gemm/benchmark --output results_run1.csv
./src/sparse_gemm/benchmark --output results_run2.csv
cp HW1_GroupN.tex HW1_Group7.tex        # poi sostituisci N con il numero del gruppo
python3 analysis/analyze.py results_run1.csv results_run2.csv -o analysis --report HW1_Group7.tex
cat analysis/summary.md
```

### 1.5 Il PDF del report

Di solito sul Pi `pdflatex` non è installato. In quel caso compila il PDF sul Mac:

```bash
# sul Mac, nella cartella hw1/
scp -r pi@raspberrypi.local:~/hw1/analysis .
scp pi@raspberrypi.local:~/hw1/results_run*.csv .
scp pi@raspberrypi.local:~/hw1/HW1_Group7.tex .     # creato da run_pi.sh con GROUP=7
pdflatex HW1_Group7.tex && pdflatex HW1_Group7.tex
```

I numeri della tabella sono scritti **dentro il `.tex`**, tra le righe `% BEGIN RESULTS` e `% END RESULTS`. Il `.tex` funziona quindi anche da solo, ad esempio su Overleaf. I grafici invece vanno caricati insieme, nella cartella `analysis/`. La colonna OI è già compilata.

**Hai già i CSV ma la tabella è vuota?** Non serve rifare i benchmark. Dalla cartella `hw1/` basta:

```bash
python3 analysis/analyze.py results_run1.csv results_run2.csv -o analysis --report HW1_Group7.tex
```

**Da completare a mano** in `HW1_Group<n>.tex`: nomi, cognomi e matricole (`Surname Name (sXXXXXX)`). Il numero del gruppo lo inserisce già `run_pi.sh`. Se non usi lo script, copia `HW1_GroupN.tex` in `HW1_Group<n>.tex` e sostituisci a mano `N` con il numero del gruppo.

### 1.6 Consigli per misure pulite

- Durante il benchmark chiudi gli altri programmi (browser, VS Code remoto, …) e non usare il Pi.
- Usa un alimentatore adeguato e, se possibile, un dissipatore. Con il calore il Pi riduce la frequenza e i tempi peggiorano. Puoi controllare così:
  ```bash
  vcgencmd measure_temp        # oltre ~80 °C il Pi rallenta
  vcgencmd get_throttled       # 0x0 = nessun rallentamento
  ```
- Facoltativo: fissa la CPU alla frequenza massima durante le misure.
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
- Lascia il default `-O0` del Makefile: è il livello di ottimizzazione previsto dal corso. Non cambiare i flag.

---

## Parte 2 – Capire il progetto

### 2.1 Lo scopo generale

Le reti neurali passano quasi tutto il tempo a moltiplicare matrici (GEMM): **C = A × W**, dove A contiene gli input e W i pesi.

Il **pruning** è una tecnica per alleggerire le reti: si azzerano i pesi meno importanti. In questo homework W (512×512) è divisa in blocchi 4×4 e solo il **25%** dei blocchi è stato tenuto (4096 su 16384). Gli altri sono tutti zero.

In teoria quindi ci sono 4× meno moltiplicazioni da fare. Ma un kernel **denso** non lo sa e moltiplica anche gli zeri. Per guadagnare davvero serve un kernel **sparso**, che salta i blocchi vuoti.

**La domanda dell'homework:** meno operazioni significano davvero meno tempo? Non per forza. Il kernel sparso:
- deve leggere gli indici per sapere dove sono i blocchi;
- accede alla memoria "a salti";
- fa meno operazioni per byte letto (OI 20.9 contro 42.7 FLOP/byte).

Infatti lo *sparse scalar* può risultare più lento del *denso ottimizzato*, pur facendo 4× meno conti. Per battere il denso bisogna ottimizzare anche lo sparso, con le tre tecniche viste a lezione:

| Tecnica | Idea | Kernel | Punti |
|---|---|---|---|
| **SIMD (NEON)** | una sola istruzione elabora 4 numeri insieme | `sparse_neon.c` | 1.0 |
| **Tiling** | riusare i dati già caricati invece di rileggerli | `sparse_tiled.c` | 1.5 |
| **Parallelismo (OpenMP)** | dividere il lavoro tra i 4 core del Pi | `sparse_openmp.c` | 1.5 |
| **Tutte insieme** | tiling + NEON + OpenMP | `sparse_openmp_tiled_neon.c` | 2.0 |

Alla fine misuri i tempi e li confronti con la teoria (FLOP e OI): questo è il report.

### 2.2 Il formato dei pesi sparsi (BCC)

Si salvano solo i blocchi non nulli, in tre array:

| Array | A cosa serve |
|---|---|
| `block_col_ptr[g] .. block_col_ptr[g+1]` | l'intervallo dei blocchi salvati per il gruppo di colonne g (colonne 4g..4g+3) |
| `block_row_idx[p]` | la riga a blocchi q del blocco p, che corrisponde alle righe 4q..4q+3 di W e agli elementi 4q..4q+3 della riga di A |
| `block_values[16p + 4r + col]` | i 16 valori del blocco p, salvati riga per riga |

Il ciclo base è in `sparse_scalar.c`:

```c
for (i = 0; i < m; ++i)                       // ogni riga di output
  for (g = 0; g < n/4; ++g) {                 // ogni gruppo di 4 colonne
    float acc[4] = {0};
    for (p = col_ptr[g]; p < col_ptr[g+1]; ++p) {   // solo i blocchi salvati
      k0 = 4 * row_idx[p];
      for (r = 0; r < 4; ++r) {               // 4 righe del blocco
        float av = a[i*k + k0 + r];
        for (col = 0; col < 4; ++col)         // 4 colonne del blocco
          acc[col] += av * values[16*p + 4*r + col];
      }
    }
    // scrive acc[0..3] in C[i, 4g..4g+3]
  }
```

### 2.3 Cosa sono `vdupq_n_f32`, `vld1q_f32`, `vfmaq_f32`, `vst1q_f32`

Il processore del Pi (ARM Cortex-A72) ha dei **registri vettoriali NEON** da 128 bit, cioè **4 float affiancati** (le 4 "lane"). In C questi registri si usano con il tipo `float32x4_t`. Le funzioni `v...` sono **intrinsics**: funzioni C che il compilatore traduce in una singola istruzione NEON.

Il nome si legge così: `v` = vettoriale, `q` = registro da 128 bit (4 float), `_f32` = float a 32 bit.

| Intrinsic | Cosa fa | Esempio |
|---|---|---|
| `vdupq_n_f32(x)` | *duplicate*: copia lo stesso numero in tutte e 4 le lane | `vdupq_n_f32(3)` → `[3,3,3,3]` |
| `vld1q_f32(ptr)` | *load*: carica 4 float consecutivi dalla memoria | → `[w0,w1,w2,w3]` |
| `vfmaq_f32(acc,a,b)` | *fused multiply-add*: `acc + a*b` lane per lane, in un'istruzione | → `[acc0+a0·b0, …, acc3+a3·b3]` |
| `vst1q_f32(ptr,v)` | *store*: scrive i 4 float in memoria | |
| `vfmaq_n_f32(acc,w,x)` | come `vfmaq_f32(acc, vdupq_n_f32(x), w)`: moltiplica i 4 pesi per lo scalare `x` e somma, senza il dup separato | |

**Perché servono qui.** Nel kernel scalare lo **stesso** valore `av` viene moltiplicato per **4 pesi adiacenti** e sommato in **4 accumulatori adiacenti**. È esattamente la forma che NEON calcola in un colpo solo:

```c
acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[0]), vld1q_f32(w_blk + 0));
```

Con l'esempio della consegna (A₀ = 1, prima riga di B₀₀ = 1 2 3 4):

```
vdupq_n_f32(A0)   → [ 1, 1, 1, 1 ]    stesso valore in 4 lane
vld1q_f32(w_blk)  → [ 1, 2, 3, 4 ]    4 pesi della riga 0 del blocco
acc               → [ 0, 0, 0, 0 ]
vfmaq_f32(...)    → [ 1, 2, 3, 4 ]    acc + A0·pesi, tutte e 4 insieme
```

Lo si ripete per le righe 1, 2 e 3 del blocco e poi per tutti i blocchi del gruppo. Alla fine `acc` contiene `[C0 C1 C2 C3]` = `[708 744 780 816]`, come nel calcolo a mano della consegna. `vst1q_f32` li scrive in C.

In pratica, ogni volta un'istruzione sostituisce 4 moltiplicazioni e 4 somme.

### 2.4 I quattro kernel implementati

**`sparse_neon.c`**: è il kernel scalare in cui il ciclo `col` (4 colonne) diventa una sola `vfmaq_f32`. C'è un accumulatore `float32x4_t` per ogni (riga, gruppo) e le 4 righe del blocco sono scritte esplicitamente una dopo l'altra.

**`sparse_tiled.c`** (senza NEON): ogni blocco di pesi serve a tutte le 128 righe di A. Il kernel base lo rilegge dalla memoria 128 volte. Qui si lavora su **4 righe di A insieme**:
- i 4 pesi di una riga del blocco si caricano **una volta** in variabili locali (`w0..w3`) e si usano per tutte e 4 le righe;
- gli accumulatori sono `acc[4][4]`, cioè il pezzo 4×4 di C;
- risultato: pesi e indici si leggono 32 volte invece di 128 (4× meno traffico).

**`sparse_openmp.c`**: le righe di output sono **indipendenti**: la riga i di C dipende solo dalla riga i di A e da W, che è in sola lettura. Quindi basta mettere `#pragma omp parallel for` sul ciclo delle righe e OpenMP divide le 128 righe tra i thread. Non servono lock perché ogni thread scrive righe diverse. Con 2 thread il tempo si dimezza circa, con 4 thread diventa circa ¼.

**`sparse_openmp_tiled_neon.c`**: unisce le tre idee.
- `#pragma omp parallel for` sul ciclo dei **blocchi di 4 righe** (32 blocchi da dividere tra i thread);
- 4 accumulatori NEON `acc0..acc3`, uno per ogni riga del blocco;
- ogni riga di pesi del blocco si carica **una volta** (`wv = vld1q_f32(...)`) e si usa per le 4 righe del blocco;
- `vfmaq_n_f32(acc_t, wv, A[t][k0+r])` aggiorna la riga t con una FMA vettore × scalare.

**Perché `vfmaq_n_f32` e non un'altra formulazione.** Le varianti sono state misurate sul Pi con GCC a `-O0` usando `experiments/try_combined_variants.sh`. La scelta dipende dal compilatore: sul Mac con clang l'ordine era diverso.

| Variante | 1 thread | 2 thread | 4 thread |
|---|---:|---:|---:|
| `vfmaq_laneq_f32` (prima versione) | 50.3 | 25.2 | 12.6 |
| `vdupq_n_f32` + `vfmaq_f32` (suggerimento del template) | 29.6 | 14.8 | 12.0 |
| **`vfmaq_n_f32`, ciclo su r (scelta)** | **26.2** | **13.2** | **6.6** |
| `vfmaq_n_f32`, righe scritte esplicitamente | 46.5 | 23.5 | 11.7 |
| `vmlaq_n_f32`, righe scritte esplicitamente | 21.8 | 11.2 | 10.8 |

Riferimento, negli stessi run: Sparse tiled 26.3 ms, Sparse NEON 39.6 ms. Tempi in ms.

A `-O0`, GCC non tiene le variabili nei registri e ogni intrinsic copia i suoi argomenti sullo stack. Più intrinsics annidati (come `laneq`, che internamente fa un dup della lane) e più variabili temporanee significano più accessi alla memoria. La variante scelta usa un solo intrinsic per ogni aggiornamento e scala in modo lineare con i thread. La V5 è più veloce con 1 thread, ma con 4 thread non migliora: probabilmente quella misura è stata disturbata, ad esempio dal Pi che si è scaldato. Per verificarlo: `VARIANTS="3 5" ./experiments/try_combined_variants.sh`.

Vincoli rispettati in tutti i kernel:
- interfacce delle funzioni e formato dei pesi invariati;
- stesso ordine di visita dei blocchi;
- solo `#pragma omp parallel for`, senza `collapse`, `schedule` o `reduction`.

### 2.5 Operational intensity (OI) – come si calcola

Si fa l'ipotesi della consegna: un solo livello di memoria e registri infiniti, quindi ogni dato si legge o scrive **una volta sola**. Float e indici occupano 4 byte.

**Denso**
- FLOP = 2·M·K·N = 2·128·512·512 = **67 108 864**
- Byte = 4·(M·K + K·N + M·N) = 4·(65 536 + 262 144 + 65 536) = **1 572 864**
- **OI = 42.7 FLOP/byte**

**Sparso**
- FLOP = 2·16·M·nnzb = 2·16·128·4096 = **16 777 216** (4× meno del denso)
- Byte:

  | Dato | Calcolo | Byte |
  |---|---|---|
  | A | 4·M·K | 262 144 |
  | valori | 4·16·nnzb | 262 144 |
  | `block_row_idx` | 4·nnzb | 16 384 |
  | `block_col_ptr` | 4·(N/4+1) | 516 |
  | C | 4·M·N | 262 144 |
  | **Totale** | | **803 332** |

- **OI = 20.9 FLOP/byte**

L'OI ideale è **la stessa per tutte le varianti sparse**. Tiling, NEON e OpenMP cambiano il traffico *effettivo* e il parallelismo, non il rapporto teorico.

### 2.6 Statistiche nel report

Il benchmark fa, per ogni kernel, 2 esecuzioni di riscaldamento e poi 5 campioni cronometrati. Nel CSV salva solo **mediana, minimo e massimo** dei 5 campioni, non i singoli valori. Per questo `analyze.py` calcola:
- **mediana** = mediana delle mediane dei vari run;
- **deviazione standard** = deviazione standard delle mediane dei vari run.

Con 2 run la std è poco affidabile: meglio usare `RUNS=5`.

### 2.7 Cosa aspettarsi (indicativo)

Sul Mac, solo per verifica (sul Pi i numeri saranno diversi, ma le tendenze simili):

| Kernel | ms | Speedup vs sparse scalar |
|---|---:|---:|
| Dense scalar | 114.9 | 0.15× |
| Dense NEON+OpenMP (4 thread) | 12.2 | 1.4× |
| Sparse scalar | 16.8 | 1.0× |
| Sparse NEON | 14.3 | 1.2× |
| Sparse tiled | 5.7 | 2.9× |
| Sparse OpenMP 4 thread | 4.6 | 3.7× |
| Sparse combined 4 thread | 1.6 | 10.2× |

Spunti per commentare i risultati nel report:
- Lo sparso fa 4× meno FLOP, ma lo *sparse scalar* non è 4× più veloce del denso ottimizzato: lo è solo rispetto al *dense scalar*.
- NEON da solo aiuta poco: il collo di bottiglia è caricare dati e indici, non fare i conti.
- Il tiling dà il guadagno più grande su un solo core, perché riduce il traffico verso la memoria.
- OpenMP scala quasi in modo lineare, perché le righe sono indipendenti.
- Combinando tutto, lo sparso batte nettamente il miglior kernel denso.

---

## Parte 3 – Problemi comuni

| Sintomo | Causa e soluzione |
|---|---|
| `HW1 requires an AArch64 ARM GCC toolchain` | Sistema a 32 bit, oppure stai compilando sul Mac. Serve Raspberry Pi OS 64-bit. |
| `FAIL sparse_...` nei test | Un kernel è stato modificato male. Ripristinalo con `git checkout -- src/sparse_gemm/<file>.c`. |
| `OpenMP setup mismatch` | Hai variabili d'ambiente OpenMP impostate. Esegui `unset OMP_NUM_THREADS OMP_DYNAMIC`. |
| `matplotlib not installed: skipping plots` | `sudo apt install python3-matplotlib`, poi rilancia solo `analyze.py`. |
| `pdflatex not found` | Normale sul Pi: compila il PDF sul Mac (sezione 1.5). |
| Tabella vuota nel `.tex` o nel PDF | Il Pi ha una versione vecchia di `analyze.py`/`run_pi.sh`: fai `git pull` (o rifai scp), poi lancia il comando `--report` della sezione 1.5. |
| `markers '% BEGIN RESULTS' ... not found` | Il `.tex` è del vecchio formato: cancellalo e rilancia `run_pi.sh`, oppure ricopialo da `HW1_GroupN.tex`. |
| Tempi molto variabili tra un run e l'altro | Pi troppo caldo o altri programmi aperti: vedi sezione 1.6. |
| `Permission denied: ./run_pi.sh` | `chmod +x run_pi.sh` |

---

## Parte 4 – Checklist di consegna

Il file da consegnare è **`HW1_Group<n>.zip`** e deve contenere:

- [ ] `HW1_Group<n>.pdf`: report compilato, con numero del gruppo, nomi e matricole
- [ ] `sparse_neon.c`
- [ ] `sparse_tiled.c`
- [ ] `sparse_openmp.c`
- [ ] `sparse_openmp_tiled_neon.c`
- [ ] `results_run1.csv`, `results_run2.csv` (i CSV grezzi del benchmark, **misurati sul Pi**)
- [ ] `analysis/summary.md` / `summary.csv` e i grafici `latency.png`, `speedup.png`, `thread_scaling.png`

Invio:
- **A:** `daniele.jahier@polito.it`
- **CC:** `valentino.peluso@polito.it`
- **Oggetto:** `ECAI27 Group<n>`

I punti di un kernel valgono solo se il kernel **passa i test** e la sua riga nel report è **compilata**.

Se il PDF è stato compilato sul Mac dopo aver eseguito `run_pi.sh`, rifai lo zip includendo il PDF:

```bash
zip -j HW1_Group7.zip HW1_Group7.pdf
```
