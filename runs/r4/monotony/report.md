## Set A: 6 Taiko-1000 test songs, human timing (R1-R3)
cells: monotone share / mean longest monotone stretch

Rhythm and colour:

| source | easy | normal | hard | oni |
|---|---|---|---|---|
| human | 0% / 2s | 2% / 2s | 1% / 4s | 0% / 0s |
| R1 (v1) | 11% / 8s | 1% / 2s | 0% / 0s | 0% / 0s |
| R2 (v2) | 1% / 4s | 2% / 6s | 0% / 0s | 0% / 0s |
| R2 all games (v3) | 3% / 8s | 1% / 5s | 0% / 0s | 0% / 0s |
| R3 best | 48% / 38s | 33% / 27s | 40% / 39s | 19% / 16s |

Rhythm only (colours ignored):

| source | easy | normal | hard | oni |
|---|---|---|---|---|
| human | 10% / 16s | 6% / 10s | 3% / 7s | 1% / 4s |
| R1 (v1) | 22% / 18s | 4% / 11s | 1% / 5s | 0% / 0s |
| R2 (v2) | 6% / 11s | 11% / 12s | 3% / 6s | 0% / 0s |
| R2 all games (v3) | 10% / 14s | 7% / 13s | 0% / 2s | 1% / 3s |
| R3 best | 51% / 40s | 36% / 33s | 52% / 48s | 66% / 60s |

## Set B: 4 benchmark songs, all courses (R4-R5)
cells: monotone share / mean longest monotone stretch

Rhythm and colour:

| source | easy | normal | hard | oni | edit |
|---|---|---|---|---|---|
| human | 1% / 6s | 0% / 0s | 1% / 8s | 0% / 2s | 0% / 0s |
| R4a oracle | 10% / 12s | 3% / 3s | 0% / 0s | 0% / 0s | 3% / 6s |
| R4a audio | 6% / 8s | 3% / 8s | 0% / 0s | 0% / 0s | 2% / 5s |
| R4 final audio | 2% / 5s | 0% / 2s | 0% / 0s | 0% / 0s | 0% / 0s |
| R5 RL audio | 3% / 9s | 0% / 0s | 0% / 0s | 0% / 0s | 0% / 0s |
| R5 RL audio x2 grid | 0% / 0s | 0% / 0s | 0% / 0s | 0% / 0s | 0% / 0s |

Rhythm only (colours ignored):

| source | easy | normal | hard | oni | edit |
|---|---|---|---|---|---|
| human | 6% / 12s | 1% / 8s | 4% / 13s | 3% / 8s | 3% / 5s |
| R4a oracle | 35% / 41s | 22% / 27s | 2% / 5s | 7% / 16s | 13% / 17s |
| R4a audio | 17% / 18s | 7% / 14s | 0% / 2s | 0% / 0s | 6% / 13s |
| R4 final audio | 20% / 23s | 4% / 14s | 3% / 6s | 22% / 26s | 41% / 38s |
| R5 RL audio | 14% / 17s | 7% / 15s | 7% / 11s | 10% / 16s | 41% / 40s |
| R5 RL audio x2 grid | 7% / 15s | 3% / 12s | 2% / 11s | 2% / 10s | 0% / 0s |

## Set C: benchmark oni/edit, 3 seeds, audio timing
cells: monotone share / mean longest monotone stretch

Rhythm and colour:

| source | oni | edit |
|---|---|---|
| R3 | 0% / 2s | 1% / 3s |
| R4a | 0% / 0s | 1% / 3s |
| R4 final | 1% / 2s | 1% / 3s |
| R5 SFT | 1% / 2s | 1% / 3s |
| R5 RL v2 | 0% / 0s | 0% / 2s |

Rhythm only (colours ignored):

| source | oni | edit |
|---|---|---|
| R3 | 29% / 26s | 14% / 20s |
| R4a | 1% / 3s | 4% / 6s |
| R4 final | 20% / 22s | 40% / 61s |
| R5 SFT | 9% / 13s | 16% / 24s |
| R5 RL v2 | 12% / 16s | 25% / 21s |

## Sensitivity (sets A+B, hard/oni/edit; human vs all generated)

| window | match | human share | generated share | human longest | generated longest |
|---|---|---|---|---|---|
| 5s | 0.85 | 4% | 7% | 6s | 5s |
| 5s | 0.9 | 3% | 5% | 5s | 4s |
| 5s | 0.95 | 2% | 3% | 3s | 3s |
| 10s | 0.85 | 1% | 5% | 3s | 5s |
| 10s | 0.9 | 1% | 4% | 3s | 4s |
| 10s | 0.95 | 0% | 2% | 1s | 2s |
| 15s | 0.85 | 0% | 5% | 2s | 6s |
| 15s | 0.9 | 0% | 3% | 0s | 4s |
| 15s | 0.95 | 0% | 2% | 0s | 2s |
