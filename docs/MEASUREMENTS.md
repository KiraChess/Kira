# Kira 1 Alpha measurements

## Internal anchored estimate

**2194.6 Elo, 95% confidence interval 2149.2–2240.0.** This is an internal Stockfish-UCI_Elo anchored estimate, not an official CCRL rating. The scale is “Elo on the Stockfish-UCI_Elo scale (nominally CCRL 40/4-anchored). Not a FIDE or Lichess rating.” No separate published-rating reference engine was available locally.

300 games at 60+0.6, 100 each against Stockfish 19 UCI_Elo 1900/2000/2100. Each engine used one thread and 64 MB hash. Fifty opening positions per anchor were played twice with colors reversed. Resignation was 1,000 cp for five plies; draw adjudication was ±10 cp for six plies after move 40.

| Stockfish UCI_Elo | W-D-L | Score | Per-anchor estimate | 95% interval |
|---:|:---:|---:|---:|:---|
| 1900 | 77-2-21 | 78.0% | 2119.9 | 2037.7–2202.1 |
| 2000 | 73-6-21 | 76.0% | 2200.2 | 2120.5–2280.0 |
| 2100 | 68-5-27 | 70.5% | 2251.3 | 2176.7–2326.0 |

The combined estimate uses inverse-variance weighting. Its interval expresses sampling uncertainty and does not cover calibration or adjudication bias. This cleanup changes author/package metadata only; the recorded games were played by the original Alpha executable with the same engine logic and NNUE weights.

## PV/adjudication caveat

fastchess recorded 5,289 Stockfish-named PV/bestmove warnings during the ladder. The warned moves match the recorded Stockfish moves, but raw UCI transcripts are absent and possible score-adjudication effects on 207 ladder games remain unquantified. The cause is not established instance by instance.

The stability soak recorded 12,085 such warnings in the Stockfish groups, with 419 of those 750 games ending by score adjudication. These warnings remain an unresolved measurement limitation.

## Relative promotion and stability

The promotion SPRTs accepted H1 against Gen1 after 214 completed games (146-26-42), and against Kira-HCE after 188 completed games (152-15-21). Two unfinished started games at each early stop are excluded from those totals. These are relative results.

The 1,000-game 5+0.05 soak used one thread, 16 MB hash, and a mix of Stockfish levels and Kira-HCE. It finished 838-32-130 (85.4% score), with no recorded technical failures caused by Kira. The soak is a stability/matchup check and is not an Elo estimate.

## Inference checks

Quantized parity covered 100,000 positions with mean absolute difference 0.05168 cp and maximum difference 1 cp. These are the existing Alpha results. The network weights remain unchanged.
