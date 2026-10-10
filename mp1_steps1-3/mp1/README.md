# ECE 3715 Mini-Project 1: Steps 1–3: Person 1 handoff

## Status and responsibilities

- Andrew Ore (Person 1): Steps 1–3 are complete here.
- Person 2: add name and agreed responsibilities.
- Person 3: add name and agreed responsibilities.
- Steps 4–8 are not implemented in this checkpoint. This is not the complete submission.
- Team number 1 is temporary. Replace `SEED = 1` near the top of mp1.ipynb with the actual team number.
- After changing the seed, restart the kernel and run all cells. Counts, tables, figures, and generated interpretations update automatically.
- Never choose a seed just to make comparisons pass. Use the actual assigned team number for submission.

## Files

- mp1.ipynb: Step 1 derivations, all 16 outcomes, one-million-packet simulation, event/intersection/union comparisons, standard errors, and Figure 1, plus the complete Step 2 Bayes derivation, three evaluations, crossover calculation, ten-million-bit validation, ratio standard error, manager explanation, and Figure 2. Step 3 includes pairwise/mutual independence derivations, one-million-pair coin simulation, conditional independence derivation, one-million-transmission receiver simulation, two printed comparison tables with standard errors, and the sensor-fusion consequence.
- mp1lib.py: required bsc(bits, p, rng) function, with a docstring and notebook tests.
- README.md: setup, responsibilities, status, runtime, and integration notes.

## Run locally

Keep these files together in the mp1 folder. With Python installed, open a terminal in this folder and run:

```sh
python -m pip install "numpy>=1.25" pandas matplotlib notebook
python -m notebook
```

On Windows, if `python` is unavailable but the Python launcher is installed, replace `python` with `py` in both commands. Open mp1.ipynb in the browser interface. Restart the kernel and run all cells from the top. NumPy 1.25 or newer is required for Generator.spawn.

Official Notebook installation and launch reference: https://jupyter.org/install

## Checkpoint validation

All 17 code cells ran from the top in a fresh Python namespace. Step 1 BSC tests, normalization, exact check values, and empirical inclusion–exclusion checks passed. Step 2 used exactly 10,000,000 bits, checked flag accounting, and reported the posterior difference and ratio SE. Step 3 used exactly 1,000,000 coin pairs and 1,000,000 transmissions. Each group with a fixed transmitted bit had more than 100,000 samples. The Pearson calculation was checked against NumPy's corrcoef. Every requested comparison reports its numerical difference and standard error. All differences between theory and simulation are within three SE for the temporary seed 1. The triple joint probability and product have different exact values, showing that mutual independence fails.

Step 3 standard errors use first-order delta-method contributions. The coin gap SE includes the effect of calculating the product and joint from the same samples. The correlation SE works for binary outputs without assuming a bivariate normal distribution. The formulas and their intermediate steps are explained in the notebook.

Measured Steps 1–3 runtime here: 0.959 seconds. Andrew also reported a successful Windows laptop run of 1.653 seconds before this wording edit. This does not measure the final eight-step notebook. The full project must run under five minutes. Record the final total after adding all eight steps.

## Integrating the later steps

The setup reserves all eight generators in order using rng.spawn(1)[0]. Step 2 should use rng_steps[1], Step 3 rng_steps[2], and so on. Create them once in the top setup cell and do not reseed inside a simulation. A separate generator is reserved for tests. Restart and run all cells for reproducible complete results.

Add the remaining required functions to mp1lib.py as their steps are implemented: packet_errors, attempts_until_clean, and tv_distance. Give each a docstring and a hand-checkable notebook test. Preserve bsc's current signature.
