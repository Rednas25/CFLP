# Sprawozdanie — CFLP Solver

## Problem

**CFLP (Capacitated Facility Location Problem)** — optymalne rozmieszczenie magazynów i przypisanie klientów przy ograniczeniach pojemnościowych. Cel: minimalizacja kosztów otwarcia magazynów + kosztów obsługi klientów. Problem NP-trudny.

**Instancje testowe** (klasyczne benchmarki):

| Instancja     | Magazyny | Klienci |
|---|---|---|
| cap41_ss.txt  | 41       | 50      |
| cap81_ss.txt  | 81       | 50      |
| cap111_ss.txt | 111      | 50      |

---

## Co zostało zaimplementowane

Wszystko w C++, bez Pythona ani zewnętrznych bibliotek.

**`CFLP.cpp`** — główny solver, rozszerzony o dwa tryby wsadowe:
- `CFLP.exe` — tryb interaktywny (menu)
- `CFLP.exe --batch` — uruchamia wszystkie algorytmy na wszystkich instancjach + sweep parametrów
- `CFLP.exe --tune` — random search tuning: 100 konfiguracji × 4 algorytmy, potem 30 finałowych runów

**`plot_results.cpp`** — generator wykresów SVG (klasa `Svg` pisząca XML wprost, bez bibliotek). Generuje 16 plików SVG w `plots/`.

**Algorytmy:**

| # | Algorytm | Opis |
|---|---|---|
| 1 | **Random** | 10 000 losowych rozwiązań |
| 2 | **Greedy** | Zachłanne przypisanie (min koszt krańcowy) |
| 3 | **EA** | Algorytm ewolucyjny: selekcja turniejowa, krzyżowanie jednorodne, mutacja |
| 4 | **SA** | Symulowane wyżarzanie |
| 5 | **VNS** | Variable Neighborhood Search: 3 struktury sąsiedztwa |
| 6 | **GRASP** | Greedy Randomized Adaptive Search + local search |

Wszystkie algorytmy stochastyczne używają `repair_solution()` gdy rozwiązanie jest niedopuszczalne.

---

## Eksperyment 1 — Baseline (domyślne parametry, 10 runów)

Domyślne konfiguracje:
- EA: pop=40, gen=250, tour=7, pm=0.90, px=0.85
- SA: T₀=180 000, cool=0.995, ipt=5
- VNS: iter=50, shake=5, ls=60
- GRASP: iter=100, rcl=2, ls=100

### Najlepsza wartość (best z 10 runów)

| Algorytm    | cap41_ss    | cap81_ss    | cap111_ss   |
|---|---|---|---|
| Random      | 2 192 350   | 2 035 884   | 2 465 973   |
| Greedy      | 1 684 846   | 1 256 212   | 1 221 731   |
| **EA**      | **1 562 334**| 1 186 398  | 1 179 712   |
| **SA**      | 1 565 048   | **1 184 874**| **1 173 848**|
| VNS         | 1 577 815   | 1 198 350   | 1 220 884   |
| GRASP       | 1 644 067   | 1 211 981   | 1 201 187   |

### Średnia (avg z 10 runów)

| Algorytm | cap41_ss  | cap81_ss  | cap111_ss |
|---|---|---|---|
| EA       | 1 578 798 | 1 190 522 | 1 192 030 |
| SA       | 1 578 259 | 1 190 703 | 1 188 290 |
| VNS      | 1 609 918 | 1 217 742 | 1 244 868 |
| GRASP    | 1 665 536 | 1 232 598 | 1 220 101 |

**Wnioski:** EA i SA są wyraźnie lepsze od VNS i GRASP. Random ok. 40% gorszy od najlepszych. Greedy daje przyzwoity wynik bez żadnej stochastyczności.

---

## Eksperyment 2 — Sweep parametrów one-at-a-time (cap41_ss, 10 runów)

Każdy parametr testowany w 3 wariantach (mniejszy / domyślny / większy), pozostałe bez zmian.

### EA

| Config   | best      | avg       |
|---|---|---|
| pop20    | 1 569 459 | 1 612 015 |
| default  | 1 562 334 | 1 578 798 |
| pop80    | 1 565 828 | 1 571 323 |
| gen100   | 1 573 615 | 1 604 159 |
| gen500   | 1 562 094 | 1 571 242 |
| pm050    | 1 567 958 | 1 585 409 |
| pm070    | 1 567 369 | 1 588 916 |
| tour3    | 1 575 966 | 1 590 090 |
| tour10   | 1 566 427 | 1 593 884 |

### SA

| Config   | best      | avg       |
|---|---|---|
| t0_50k   | 1 562 715 | 1 585 512 |
| default  | 1 565 048 | 1 578 259 |
| t0_500k  | 1 562 561 | 1 572 755 |
| cool990  | 1 565 893 | 1 596 522 |
| cool999  | 1 559 754 | **1 563 833** |
| iter2    | 1 575 670 | 1 608 456 |
| iter10   | 1 560 589 | 1 567 136 |

### VNS

| Config   | best      | avg       |
|---|---|---|
| iter20   | 1 594 726 | 1 629 814 |
| default  | 1 577 815 | 1 609 918 |
| iter100  | 1 571 263 | 1 585 623 |
| shake3   | 1 579 343 | 1 601 545 |
| shake8   | 1 577 002 | 1 610 092 |
| ls30     | 1 602 339 | 1 660 576 |
| ls120    | 1 565 950 | **1 585 064** |

### GRASP

| Config   | best      | avg       |
|---|---|---|
| iter50   | 1 660 266 | 1 676 491 |
| default  | 1 644 067 | 1 665 536 |
| iter200  | 1 638 086 | 1 653 351 |
| rcl1     | 1 633 243 | **1 641 913** |
| rcl4     | 1 711 008 | 1 743 563 |
| ls50     | 1 631 988 | 1 668 272 |
| ls200    | 1 612 811 | 1 635 031 |

**Wnioski sweep:**
- SA: wolne chłodzenie (cool=0.999) daje wyraźną poprawę
- VNS: więcej ls_attempts ważniejsze niż więcej iteracji czy shake'ów
- GRASP: rcl=1 (czysto zachłanne konstruowanie) lepsze niż rcl=4 (zbyt dużo losowości)
- EA: większy pop i gen pomagają, tour_size ma mniejszy wpływ

---

## Eksperyment 3 — Random Search Tuning

### Metodologia

Zamiast testowania parametrów jeden po drugim (co faworyzuje domyślne wartości pozostałych), wszystkie parametry są losowane **jednocześnie** — to sprawiedliwe porównanie przestrzeni parametrów.

- **Faza selekcji:** 100 losowych konfiguracji × 3 runy na cap41_ss
- **Faza finalna:** najlepsza konfiguracja × **30 runów** na wszystkich 3 instancjach
- Seed = 42 (reprodukowalne)

### Zakresy próbkowania

| EA | SA | VNS | GRASP |
|---|---|---|---|
| pop ∈ [10,100] | T₀ ∈ [5k,300k] | iter ∈ [10,200] | iter ∈ [20,300] |
| gen ∈ [50,600] | cool ∈ [0.98,0.999] | shake ∈ [1,15] | rcl ∈ [1,8] |
| tour ∈ [2,15] | min_T ∈ [0.5,10] | ls ∈ [10,200] | ls ∈ [20,300] |
| pm ∈ [0.1,1.0] | ipt ∈ [1,10] | | |
| px ∈ [0.5,1.0] | | | |
| bias ∈ [0.3,0.8] | | | |

### Najlepsze znalezione konfiguracje

**EA** (config #40): pop=95, gen=430, tour=7, pm=0.566, px=0.919, bias=0.638 → avg(3 runy)=1 564 955

**SA** (config #65): T₀=210 604, cool=0.99895, min_T=9.02, ipt=4 → avg=1 562 086

**VNS** (config #99): iter=181, shake=4, ls=176 → avg=1 565 650

**GRASP** (config #49): iter=279, rcl=1, ls=292 → avg=1 601 520

### Wyniki finalne — 30 runów (best / avg / std / czas ms)

| Algorytm | cap41_ss | cap81_ss | cap111_ss |
|---|---|---|---|
| **EA**    | 1 559 750 / 1 569 400 / 6 808 / 1554 ms | 1 184 370 / 1 193 274 / 5 099 / 1508 ms | 1 173 870 / 1 189 153 / 7 924 / 1575 ms |
| **SA**    | 1 559 750 / 1 566 278 / 6 669 / 2410 ms | 1 180 740 / 1 187 747 / 3 439 / 2310 ms | **1 167 030 / 1 177 576** / 6 485 / 2310 ms |
| **VNS**   | **1 559 750 / 1 565 759** / 5 448 / 1276 ms | 1 180 960 / 1 188 120 / 3 777 / 1210 ms | 1 176 040 / 1 194 492 / 8 759 / 1392 ms |
| GRASP     | 1 588 660 / 1 604 555 / 6 303 / 792 ms  | 1 191 150 / 1 197 534 / 3 914 / 701 ms  | 1 184 550 / 1 187 815 / 1 888 / 765 ms  |

*Format: best / avg / std / czas. Niższe = lepsze.*

---

## Wnioski końcowe

### Ranking po tuningu (avg na cap41_ss)

1. **VNS** — 1 565 759
2. **SA** — 1 566 278
3. **EA** — 1 569 400
4. GRASP — 1 604 555

### Kluczowe obserwacje

1. **SA dominuje na większych instancjach** (cap111): najniższa avg i mała std — najbardziej niezawodny
2. **VNS najlepszy na cap41** — skuteczny local search + umiarkowana złożoność; gorsza stabilność na cap111
3. **GRASP wyraźnie słabszy** nawet po tuningu (~2% gorszy od SA/VNS); najlepsza konfiguracja to rcl=1, co oznacza że losowość w fazie konstruowania przeszkadza
4. **EA konkurencyjny**, ale wymaga więcej czasu; większa wariancja niż SA
5. **Spread 100 konfiguracji**: SA i VNS mniej wrażliwe na złe parametry niż EA (EA z pop=10 potrafi być 10% gorszy od najlepszego)

| Algorytm | Najważniejszy parametr | Wniosek |
|---|---|---|
| EA | gen, pop_size | Duży budżet obliczeniowy; tour_size mniej ważny |
| SA | cooling_rate | Wolne chłodzenie (>0.998) kluczowe |
| VNS | local_search_attempts | Więcej LS > więcej shake'ów |
| GRASP | rcl_size | rcl=1 (czysto zachłanne); LS ważniejszy niż liczba iteracji |

---

## Struktura plików

```
CFLP.cpp                  -- solver (--batch, --tune)
plot_results.cpp          -- generator SVG
data/
  batch_summary.csv       -- wyniki baseline + sweep
  batch_log.txt           -- log z --batch
  tune_log.txt            -- log z --tune
  tuning/
    tuning_EA.csv         -- 100 konfiguracji EA z wynikami (3 runy każda)
    tuning_SA.csv
    tuning_VNS.csv
    tuning_GRASP.csv
    final_results.csv     -- 30-runowe wyniki najlepszych konfiguracji
  ea_default_cap41_ss/    -- CSV postępu per generacja (każdy run osobno)
  sa_default_cap41_ss/    -- CSV postępu SA
  vns_default_cap41_ss/
  grasp_default_cap41_ss/
plots/
  01_baseline_comparison.svg       -- wszystkie algorytmy x instancje
  02_ea_convergence.svg            -- zbieżność EA (avg 10 runów)
  03_sa_convergence.svg
  04_vns_convergence.svg
  05_grasp_convergence.svg
  06_ea_param_sweep.svg            -- wrażliwość na parametry (one-at-a-time)
  07_sa_param_sweep.svg
  08_vns_param_sweep.svg
  09_grasp_param_sweep.svg
  10_convergence_all_algorithms.svg
  11_scalability.svg               -- jakość vs rozmiar instancji
  tuning_EA_ranked.svg             -- 100 konfiguracji posortowanych (best=czerwony)
  tuning_SA_ranked.svg
  tuning_VNS_ranked.svg
  tuning_GRASP_ranked.svg
  tuning_final_comparison.svg      -- tuned best x instancje
```
