#!/usr/bin/env bash
export PYTHONPATH="$(pwd)"
python scripts/run_all_phases.py
python scripts/run_phase4.py
python scripts/run_noise_sweep.py --phase prompt
echo "All phases complete. See results/."
