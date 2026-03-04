# Demo Folder (Public-Safe)

This folder keeps small, shareable examples that represent the core technical ideas without exposing private training/data assets.

## Files

- `emit_bandpass_simulation_demo.py`
  - Demonstrates EMIT-like hyperspectral -> target multispectral conversion using SRF interpolation + bandpass integration + percentile alignment.
- `multidomain_vit_adapter_demo.py`
  - Demonstrates a shared backbone with domain-specific low-rank adapters and sensor-specific heads.

## Run

```bash
python3 portfolio/demo/emit_bandpass_simulation_demo.py
python3 portfolio/demo/multidomain_vit_adapter_demo.py
```

Both scripts are toy examples with synthetic tensors only.
