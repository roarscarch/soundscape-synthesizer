# Soundscape Synthesizer

Takes a phrase like "forest at night" and generates an endless ambient soundscape. Uses wave function collapse to keep it from repeating.

## Why?
I wanted background noise while coding that wouldn't loop predictably. Music has structure, white noise is boring - this sits in between.

## Usage
```
pip install -r requirements.txt
python src/main.py --seed "ocean waves"
```

## Biomes
- Forest (birds, wind, leaves)
- Ocean (waves, distant storms)
- Space (low drones, occasional events)
- Custom: drop your own wavetables in `samples/`

