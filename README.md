# ECE 3715 Mini-Project 1 — Team 1

Probability, Bayes, and discrete random variables, applied to a link that sends packets of bits and retransmits corrupted ones until they get through.

## Team members and contributions

| Member | What they did |
|---|---|
| Andrew Ore | Steps 1–3: the sample space and the axioms, conditional probability and Bayes, independence (Figures 1–2), and `bsc` |
| Trevor Cheng | Steps 4–5: Bernoulli trials and the Binomial, the Poisson approximation (Figures 3–5) |
| George Li | Steps 6–8: retransmissions and the geometric distribution, how fast estimates settle, the throughput of the link (Figures 6–9), and `attempts_until_clean` |

## Total runtime

About **10 seconds** for the whole notebook on a clean kernel (Kernel → Restart Kernel and Run All Cells), well under the 5-minute limit. The last cell of `mp1.ipynb` prints the exact runtime on whatever machine runs it.

## Contents

| File | What it is |
|---|---|
| `mp1.ipynb` | The notebook: Steps 1–8, with a summary of every check-yourself value at the end. Runs top to bottom. |
| `mp1lib.py` | The four required functions: `bsc`, `packet_errors`, `tv_distance`, `attempts_until_clean`. Each has a docstring and a hand-checkable test near the top of the notebook. |
| `README.md` | This file. |

## How to run

Requires Python 3 with NumPy 1.25 or newer, SciPy, Matplotlib, pandas and Jupyter. Keep the three files in the same folder, open `mp1.ipynb`, and run **Kernel → Restart Kernel and Run All Cells**.

Team seed used: 1
