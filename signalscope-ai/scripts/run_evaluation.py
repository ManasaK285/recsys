import json
from pathlib import Path
from signalscope.simulator import generate_telemetry
from signalscope.storage import TelemetryStore
from signalscope.analytics import evaluation_metrics

def main():
    store = TelemetryStore()
    if store.count() == 0:
        store.replace(generate_telemetry())
    result = evaluation_metrics(store.read())
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved outputs/evaluation.json")

if __name__ == "__main__":
    main()
