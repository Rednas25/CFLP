# Sprawozdanie — Eksperymenty z budzetem obliczeniowym

**Instancja:** cap41_ss.txt (41 magazynow, 50 klientow)

## Metodologia

Kazdy algorytm dostaje **rowny budzet = 100000 ewaluacji** na jeden run.
Budzet jest egzekwowany inaczej dla kazdego algorytmu:

| Algorytm | Mechanizm budgetu |
|---|---|
| EA | `pop * (1 + gen) = 100000` — gen wyliczany z pop |
| SA | licznik ewaluacji; stop gdy `evals >= 100000` |
| VNS | `iter * (shake + ls) ≈ 100000` — ls wyliczane z iter+shake |
| GRASP | `iter * (1 + ls) ≈ 100000` — ls wyliczane z iter |

**Konfiguracje:** 100 losowych (seed=77)
**Runy na konfiguracj:** 10 (bez usredniania — kazdy run osobno w CSV)

---

## Wyniki EA (100 konfiguracji)

Zakres: pop ∈ [5,300], gen=budget/pop-1, tour ∈ [2,15], pm ∈ [0.1,0.95], px ∈ [0.5,0.95], bias ∈ [0.3,0.8]

### Top-10 konfiguracji EA (po avg)

| Rank | Config | Parametry | best | avg | std |
|---|---|---|---|---|---|
| 1 | #64 | pop=234 gen=426 pm=0.94 | 1559754 | 1561487 | 2423 |
| 2 | #26 | pop=171 gen=583 pm=0.83 | 1559754 | 1561523 | 2364 |
| 3 | #57 | pop=163 gen=612 pm=0.88 | 1559754 | 1562780 | 3218 |
| 4 | #68 | pop=282 gen=353 pm=0.80 | 1559754 | 1562892 | 3165 |
| 5 | #20 | pop=157 gen=635 pm=0.90 | 1559754 | 1562999 | 2863 |
| 6 | #37 | pop=292 gen=341 pm=0.86 | 1559754 | 1563481 | 4683 |
| 7 | #43 | pop=299 gen=333 pm=0.67 | 1562715 | 1563755 | 1135 |
| 8 | #28 | pop=9 gen=11110 pm=0.76 | 1559754 | 1563763 | 4633 |
| 9 | #90 | pop=78 gen=1281 pm=0.80 | 1559754 | 1563772 | 4255 |
| 10 | #3 | pop=76 gen=1314 pm=0.84 | 1559754 | 1563936 | 4570 |

**Najgorsza konfiguracja:** #73 (pop=47 gen=2126 pm=0.22) avg=1593058, std=22735

Rozrzut (best avg vs worst avg): 31571

---

## Wyniki SA (100 konfiguracji)

Zakres: T0 ∈ [500,800k], cool ∈ [0.90,0.9999], min_T ∈ [0.1,20], ipt ∈ [1,20]

### Top-10 konfiguracji SA (po avg)

| Rank | Config | Parametry | best | avg | std |
|---|---|---|---|---|---|
| 1 | #29 | T0=166130 cool=0.9983 ipt=17 | 1559754 | 1560788 | 1553 |
| 2 | #20 | T0=199602 cool=0.9991 ipt=8 | 1559754 | 1561132 | 1719 |
| 3 | #15 | T0=610837 cool=0.9990 ipt=6 | 1559754 | 1561181 | 1931 |
| 4 | #64 | T0=643624 cool=0.9987 ipt=16 | 1559754 | 1562229 | 2139 |
| 5 | #4 | T0=225904 cool=0.9973 ipt=15 | 1559754 | 1562281 | 1747 |
| 6 | #33 | T0=285978 cool=0.9851 ipt=19 | 1559754 | 1567048 | 6670 |
| 7 | #80 | T0=210264 cool=0.9846 ipt=16 | 1566881 | 1574258 | 10090 |
| 8 | #1 | T0=562141 cool=0.9889 ipt=14 | 1563500 | 1575045 | 9804 |
| 9 | #8 | T0=527541 cool=0.9895 ipt=14 | 1559754 | 1577208 | 17553 |
| 10 | #2 | T0=465116 cool=0.9870 ipt=10 | 1565051 | 1578814 | 8437 |

**Najgorsza konfiguracja:** #66 (T0=767352 cool=0.9209 ipt=1) avg=2422624, std=422898

Rozrzut: 861836

---

## Wyniki VNS (100 konfiguracji)

Zakres: iter ∈ [1,1000], shake ∈ [1,15]; ls = budget/iter - shake

### Top-10 konfiguracji VNS (po avg)

| Rank | Config | Parametry | best | avg | std |
|---|---|---|---|---|---|
| 1 | #29 | iter=76 shake=1 ls=1314 | 1559754 | 1559754 | 0 |
| 2 | #35 | iter=35 shake=2 ls=2855 | 1559754 | 1560123 | 747 |
| 3 | #87 | iter=176 shake=3 ls=565 | 1559754 | 1560543 | 1273 |
| 4 | #41 | iter=233 shake=2 ls=427 | 1559754 | 1560564 | 1405 |
| 5 | #30 | iter=55 shake=15 ls=1803 | 1559754 | 1560635 | 2301 |
| 6 | #6 | iter=98 shake=12 ls=1008 | 1559754 | 1560715 | 1234 |
| 7 | #10 | iter=24 shake=15 ls=4151 | 1559754 | 1560756 | 1280 |
| 8 | #49 | iter=16 shake=2 ls=6248 | 1559754 | 1560780 | 1378 |
| 9 | #32 | iter=60 shake=9 ls=1657 | 1559754 | 1560960 | 1540 |
| 10 | #27 | iter=137 shake=4 ls=725 | 1559754 | 1560961 | 1297 |

**Najgorsza konfiguracja:** #2 (iter=1000 shake=11 ls=89) avg=1571063, std=15765

Rozrzut: 11309

---

## Wyniki GRASP (100 konfiguracji)

Zakres: iter ∈ [1,1000], rcl ∈ [1,6]; ls = budget/iter - 1

### Top-10 konfiguracji GRASP (po avg)

| Rank | Config | Parametry | best | avg | std |
|---|---|---|---|---|---|
| 1 | #100 | iter=29 rcl=2 ls=3447 | 1561851 | 1567428 | 3820 |
| 2 | #13 | iter=87 rcl=1 ls=1148 | 1568803 | 1574086 | 2742 |
| 3 | #96 | iter=7 rcl=3 ls=14284 | 1564710 | 1575867 | 9432 |
| 4 | #27 | iter=45 rcl=5 ls=2221 | 1570841 | 1578791 | 4626 |
| 5 | #9 | iter=172 rcl=1 ls=580 | 1574202 | 1584092 | 6177 |
| 6 | #50 | iter=106 rcl=3 ls=942 | 1581845 | 1590079 | 7355 |
| 7 | #47 | iter=232 rcl=1 ls=430 | 1580962 | 1592951 | 8288 |
| 8 | #31 | iter=259 rcl=1 ls=385 | 1588726 | 1593402 | 3160 |
| 9 | #28 | iter=232 rcl=1 ls=430 | 1579686 | 1593813 | 8464 |
| 10 | #6 | iter=133 rcl=2 ls=750 | 1585914 | 1594814 | 6533 |

**Najgorsza konfiguracja:** #53 (iter=874 rcl=6 ls=113) avg=1745727, std=18677

Rozrzut: 178299

---

## Porownanie najlepszych konfiguracji (fair budget=100k)

| Algorytm | Najlepsza konfiguracja | best | avg | std | rozrzut 100 konfiguracji |
|---|---|---|---|---|---|
| **EA** | #64 pop=234 gen=426 pm=0.94 | 1559754 | 1561487 | 2423 | 31571 |
| **SA** | #29 T0=166130 cool=0.9983 ipt=17 | 1559754 | 1560788 | 1553 | 861836 |
| **VNS** | #29 iter=76 shake=1 ls=1314 | 1559754 | 1559754 | 0 | 11309 |
| **GRASP** | #100 iter=29 rcl=2 ls=3447 | 1561851 | 1567428 | 3820 | 178299 |

### Wnioski

Przy rownym budzecie 100k ewaluacji na cap41_ss:

- **VNS** osiaga najnizsze avg wsrod najlepszych konfiguracji
- Rozrzut 100 konfiguracji mierzy wrazliwosc na parametry: duzy rozrzut = algorytm trudny w strojeniu
- SA z budzetem = twardy limit ewaluacji (niezaleznie od temperatury)
- EA: duze pop (mniej generacji) vs male pop (wiecej generacji) — trade-off widoczny na wykresie pop vs avg
- VNS/GRASP: malo iteracji z glebokim LS vs duzo iteracji z plytkim LS — trade-off na wykresach iter vs avg

---

## Pliki wynikowe

```
data/budget/ea_budget.csv        -- 100 konfiguracji EA (100 runow kazdej)
data/budget/sa_budget.csv
data/budget/vns_budget.csv
data/budget/grasp_budget.csv
plots/budget/ea_pop_vs_avg.svg   -- scatter: pop_size vs avg
plots/budget/ea_pm_vs_avg.svg    -- scatter: mutation_rate vs avg
plots/budget/ea_tour_vs_avg.svg  -- scatter: tour_size vs avg
plots/budget/ea_px_vs_avg.svg    -- scatter: cross_pro vs avg
plots/budget/sa_cool_vs_avg.svg  -- scatter: cooling_rate vs avg
plots/budget/sa_T0_vs_avg.svg    -- scatter: initial_temp vs avg
plots/budget/sa_ipt_vs_avg.svg   -- scatter: iter_per_temp vs avg
plots/budget/vns_iter_vs_avg.svg -- scatter: max_iter vs avg
plots/budget/vns_ls_vs_avg.svg   -- scatter: ls_attempts vs avg
plots/budget/vns_shake_vs_avg.svg
plots/budget/grasp_iter_vs_avg.svg
plots/budget/grasp_rcl_vs_avg.svg
plots/budget/grasp_ls_vs_avg.svg
SPRAWOZDANIE_BUDGET.md           -- ten plik
```
