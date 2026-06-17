# woodcut-ai

Turn photographs (e.g. your Desolation Wilderness shots) into **laser-cuttable,
Killion-style woodblock print layers** — with Claude as the analysis and
judging backbone, classical computer vision for the laser-ready geometry, and a
pluggable diffusion slot for the "look."

> Tom Killion calls his style "faux ukiyo-e" — Hokusai/Hiroshige landscape
> composition crossed with Eric Gill / Rockwell Kent wood-engraving. Notably, he
> works from **sketches, not photos**, precisely so he can *simplify*. This tool's
> whole job is to do that reduction: collapse a photo into a few bold,
> registered, carvable blocks.

## Why this architecture

"Use AI to cut the shapes" is really three jobs, and only one is generative:

| Stage | Tool | Why |
|---|---|---|
| **Analyze & plan** the reduction (palette, layers, key block) | **Claude `claude-opus-4-8`** (vision) | High-res vision; reasons about composition and carvability the way Killion reads a sketch. |
| **Stylize** the woodblock *look* | **Diffusion model** (pluggable adapter; stub by default) | Claude doesn't generate raster art. This is the slot you benchmark. |
| **Separate → vectorize → cut files** | **Classical CV** (k-means, edges, `potrace`) — *not* generative | Deterministic and laser-ready. Diffusion output is beautiful but can't be fed to a laser directly. |
| **Judge** candidates in the benchmark | **Claude** (LLM-judge) | Scores woodblock fidelity, layer separation, and carvability; you make the final taste call from the contact sheet. |

The deliberate choice: **Claude + classical CV for everything precise, diffusion
as an optional style oracle.** Many strong results come from posterizing the
photo directly with no diffusion at all.

## The print model (from Killion's process)

- **Key block** — darkest, most detail; carved first; carries the outlines and
  the **registration** every color block aligns to; printed **last**.
- **Color blocks** — derived from the key block, printed **lightest → darkest**.
- **Two modes**: `separate` (one block per color) or `reduction` (one block
  carved progressively, overprinting — layer order is load-bearing).
- **Split-fountain** skies (one block, a gradient) and **semi-transparent
  overlays** (a few blocks → many colors) are first-class in the data model.

Every block SVG carries identical **registration ticks** — the digital
equivalent of pulling proof sheets off the key block to align each color block.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# external, for true vector cut paths (not pip):
#   macOS: brew install potrace      Debian/Ubuntu: apt-get install potrace
cp .env.example .env        # optional: add ANTHROPIC_API_KEY for real analysis/judging
```

Runs **fully offline** without a key (heuristic plan + stub stylizer + no judge),
so you can exercise the whole pipeline before wiring anything up.

## Use

```bash
# Claude's reduction of a photo into a block plan (JSON)
woodcut analyze photos/lake.jpg --layers 5

# Full pipeline -> per-block laser SVGs + a stacked color preview
woodcut make photos/lake.jpg -o outputs/lake
woodcut make photos/lake.jpg --no-stylize        # classical-CV-only path

# Sweep approaches/params over a folder, judge with Claude, build a contact sheet
woodcut bench --photos photos

woodcut adapters                                  # list stylize adapters
```

`make` writes, under the output dir:
`plan.json`, `stylized.png` (if using the stylizer), `masks/` (per-layer),
`blocks/block_NN_*.svg` (what you send to the laser), and `preview.svg`.

## Wiring a real diffusion model

The stylize slot is an adapter (`woodcut/stylize/base.py`). To add a provider:

1. Copy `woodcut/stylize/stub.py`, implement `stylize()` against your API or
   local endpoint (e.g. ComfyUI / Stable Diffusion).
2. Register it in `woodcut/stylize/__init__.py`.
3. Set `WOODCUT_STYLIZE_ADAPTER=yourname`, and add it to `benchmark/matrix.py`
   so it's swept against the others.

Nothing downstream changes — separation/vectorize/laser don't care which adapter
produced the image.

## Benchmarking

`benchmark/matrix.py` defines the experiment grid (approach × adapter × layer
count × mode). `woodcut bench` runs every cell on every photo, has Claude
**judge** each result (woodblock fidelity, shape boldness, layer separation,
carvability, Killion resemblance), and emits `results.json` + a
`contact_sheet.png` for your own taste call. Start with the CV-only baseline and
add diffusion adapters as you get access.

## Layout

```
woodcut/
  models.py      domain model (key block, color blocks, reduction, registration)
  analysis.py    Claude vision -> BlockPlan  (heuristic fallback offline)
  prompts.py     Killion-informed prompts (analysis, stylize, judge)
  stylize/       pluggable diffusion adapters (base + stub)
  separate.py    k-means color separation + key-block edge mask  (CV)
  vectorize.py   potrace raster->vector  (CV)
  laser.py       per-block SVGs + registration marks + color preview
  judge.py       Claude LLM-judge
  pipeline.py    end-to-end orchestration
benchmark/       matrix, runner, contact sheet
cli.py           woodcut CLI
```

## From file to finished print

1. `woodcut make` → block SVGs.
2. Import each `block_NN_*.svg` into your laser software; cut/engrave the
   **white** (printing) areas into shina ply / linoleum. The magenta
   registration ticks become a non-cut alignment reference (or their own light
   pass).
3. Ink lightest block first, register against the key block, print
   lightest → darkest, key block last — exactly Killion's press order.
