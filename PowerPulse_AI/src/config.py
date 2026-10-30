from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=Path(os.getenv('POWERPULSE_DATA_DIR',ROOT/'data'))
DATA_PATH=DATA_DIR/'device_telemetry.parquet'
