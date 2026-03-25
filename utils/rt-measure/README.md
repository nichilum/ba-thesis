# RT60 Measurement Utility

This script reads a `metadata.jsonl` file, generates a synthetic impulse response per item using `pedalboard.Reverb` with each line's `size` and `wetness`, and measures RT60 with `pyroomacoustics`.

The audio paths in metadata are ignored for this workflow.

## Run RT60 Summary

```bash
python main.py measure --metadata /path/to/metadata.jsonl
```

## Measure Options

- `--split {all,train,val,test}`: filter by split (default: `all`)
- `--sample-rate INT`: sample rate for synthetic impulse response (default: `44100`)
- `--ir-duration FLOAT`: synthetic IR duration in seconds (default: `3.0`)
- `--max-items INT`: optional cap for quick tests
- `--dist-plot PATH`: save RT60 distribution plot (PNG)
- `--dist-bins INT`: histogram bin count for distribution plot (default: `40`)

## Measure Example

```bash
python main.py measure \
	--metadata ../../experiments/perceptual-quality/data/metadata.jsonl \
	--split val \
	--sample-rate 44100 \
	--ir-duration 3.0 \
	--max-items 100
```

Generate a distribution plot over all measured RT60 values:

```bash
python main.py measure \
	--metadata ../../experiments/perceptual-quality/data/metadata.jsonl \
	--split all \
	--dist-plot plots/rt60_distribution_all.png
```

## Output

The script prints console stats only:

- `processed`
- `succeeded`
- `failed`
- `clipped_parameters`
- `mean`, `median`, `std`, `min`, `max` (RT60 seconds)

## Generate Plots with pyfar

This creates nice figure files from synthesized reverberated impulses:

```bash
python main.py plot \
	--metadata ../../experiments/perceptual-quality/data/metadata.jsonl \
	--split val \
	--num-plots 6 \
	--plot-dir plots/val \
	--plot-style light
```

### Plot Outputs

- `ir_XX_size_..._wet_....png`: one time/frequency plot per selected impulse
- `impulse_bank_time2d.png`: 2D time map across plotted impulses
- `impulse_representative_spectrogram.png`: spectrogram of one representative impulse

### Plot Options

- `--num-plots INT`: number of per-impulse figures (default: `6`)
- `--plot-dir PATH`: output directory for figures (default: `plots`)
- `--plot-style {light,dark}`: pyfar style for saved figures (default: `light`)
