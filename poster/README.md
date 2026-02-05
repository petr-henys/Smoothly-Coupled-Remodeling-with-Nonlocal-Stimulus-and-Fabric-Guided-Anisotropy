# LaTeX poster

## Build

From `poster/`:

```bash
latexmk -pdf main.tex
```

Output PDF: `poster/build/main.pdf`

Notes:
- `poster/latexmkrc` uses XeLaTeX (consistent fonts with the `presentation/` deck).
- Figures are referenced from `presentation/figures/` to avoid duplication.
