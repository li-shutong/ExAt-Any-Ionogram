---
name: ionogram-scaling
description: Extracts F-layer echo traces from arbitrary ionogram images using ExAt, a VLM multi-agent PDSA pipeline. Use when the user needs ionogram segmentation, trace extraction, scaling, foF2, or h'F2.
---

# ExAt — Ionogram Scaling Skill

Zero-shot F-layer first-hop trace extraction for arbitrary ionograms. Three VLM agents plus a code renderer; no station-specific weights.

## When to use

- Process, label, or segment an ionogram
- Extract F-layer echo traces (O/X)
- Compute foF2 / h'F2
- Evaluate predictions against GT
- Analyze PDSA iteration quality

## Pipeline

1. **Iono-Describer** (VLM): layers, O/X morphology A–E, exclusion notes
2. **Iono-Tracer** (VLM): pixel polylines on the F-layer first-hop ridge
3. **Iono-Critic** (VLM): segment offsets + five scores; revise if confidence < 0.9
4. **MaskBuilder** (code): rasterize the best polyline into a binary mask

Keep the highest-scoring candidate across at most 3 PDSA rounds.

## Commands

```bash
pip install -r requirements.txt
set VLM_API_KEY=your-own-key
set VLM_BASE_URL=https://api.openai.com/v1
set VLM_MODEL=gpt-4o

python run.py --image path/to/ionogram.png
python run.py --batch path/to/images/ --output-dir output/
python tools/labeling.py path/to/images/
python tools/evaluate.py --paper path/to/paper_data
```

If `VLM_API_KEY`, `VLM_BASE_URL`, or `VLM_MODEL` is missing, ask the user to set them.

## Outputs

`<output_dir>/<stem>/` contains description, `trace_iterN`, `critic_iterN`, `mask.png`, `overlay.png`, and `result.json`.
