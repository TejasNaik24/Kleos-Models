# Research paper

*Learning a Decision Policy from Synthetic Data: Pre-Registered QLoRA Fine-Tuning of Open-Weight Models for a Student AI Operating System.* Tejas Naik, The Ohio State University, October 2026.

The paper is the written account of everything in this repository: the synthetic decision data, the QLoRA recipe, the pre-registered hypotheses H1 to H9, the results for KLEOS Hermes and KLEOS Logos, the serving measurements and the limitations.

| File | What it is |
| --- | --- |
| [`kleos-decision-policy.pdf`](kleos-decision-policy.pdf) | The compiled paper. GitHub renders it in the browser. |
| [`main.tex`](main.tex) | The LaTeX entry point: preamble, title block, section inputs, bibliography. |
| [`sections/`](sections/) | One file per section (`00-abstract` to `11-conclusion`) and `appendix.tex`. |
| [`figures/`](figures/) | The five figures, drawn with TikZ and pgfplots from recorded numbers; no image files. |
| [`references.bib`](references.bib) | The bibliography. Every entry was checked against the publisher, arXiv or the vendor page it cites. |

## Where the numbers come from

Every number in the paper is copied from a frozen record in this repository or in [Kleos-Training-Data](https://github.com/TejasNaik24/Kleos-Training-Data). A `% source:` comment above each table, figure series and headline paragraph names the record and line it was copied from (`EXP` is `docs/experiments.md`, `HER`, `L1` and `L2` are the run reports under `docs/experiments/`, `SVR` the serving records, `SC` and `V7` the data pipeline's `docs/scenarios.md` and v0.0.7 design document). Nothing is recomputed, and values the records keep only privately are not used.

## Compiling

The source compiles with pdfLaTeX and BibTeX on a full TeX Live, for example on Overleaf:

1. Zip this folder (without the PDF) and upload it as a new Overleaf project, or add the files to an existing one.
2. Keep the default compiler (pdfLaTeX) and set `main.tex` as the main document.
3. Recompile; Overleaf runs BibTeX and the second pass itself.
4. Download the PDF and save it here as `kleos-decision-policy.pdf`.

Packages used: `geometry`, `newtxtext`, `newtxmath`, `microtype`, `booktabs`, `tabularx`, `multirow`, `caption`, `subcaption`, `enumitem`, `xcolor`, `tikz`, `pgfplots` (compat 1.18), `natbib` with `plainnat`, `hyperref`. All are in TeX Live.

## Citing

See the [Citation](../README.md#citation) section of the repository README, or `CITATION.cff`, which names the paper as the preferred citation.
