import argparse, json
import torch
from src.experiments.common import load_data_and_config
from src.models.sasrec import SASRec
from src.models.bt_sr import BTSR
from src.evaluation.representation import collect_representations, effective_rank
from src.utils.config import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["sasrec", "bt"], default="bt")
    args = parser.parse_args()

    cfg, mappings, train, val, test = load_data_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls = SASRec if args.model == "sasrec" else BTSR
    kwargs = dict(
        num_items=int(mappings["num_items"]),
        max_seq_len=cfg["model"]["max_seq_len"],
        d_model=cfg["model"]["embedding_dim"],
        nhead=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )
    if args.model == "bt":
        kwargs.update(alpha=cfg["barlow_twins"]["alpha"],
                      lambda_offdiag=cfg["barlow_twins"]["lambda_offdiag"])

    model = cls(**kwargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    histories = []
    for rec in test:
        hist = rec["items"][:-1][-cfg["data"]["max_seq_len"]:]
        histories.append([0] * (cfg["data"]["max_seq_len"] - len(hist)) + hist)

    z = collect_representations(model, histories, device)
    erank, singular = effective_rank(z)

    out = ROOT / cfg["paths"]["results_dir"] / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "representation.json", "w") as f:
        json.dump({"effective_rank": erank}, f, indent=2)

    torch.save(singular, out / "singular_values.pt")
    print(f"Effective rank: {erank:.3f}")

if __name__ == "__main__":
    main()
