$env:PYTHONPATH = (Get-Location).Path
python scripts/run_all_phases.py
python scripts/run_phase4.py
python scripts/run_noise_sweep.py --phase prompt
Write-Host "All phases complete. See results/."
