@echo off
python -m pip install -r requirements.txt
python run_experiment.py --iterations 24 --seed 42 --mode pareto
pause
