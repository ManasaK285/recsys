from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.utils import load_config
from src.data import generate_synthetic, encode_and_split
from src.model import ASMoE


st.set_page_config(
    page_title="ASMoE Sequential Recommendation",
    page_icon="🧠",
    layout="wide",
)
st.title("ASMoE Sequential Recommendation")
st.caption(
    "Interactive next-item recommendation demo · Synthetic sequential interactions · "
    "Metrics are for this experiment only, not real-world performance."
)

cfg = load_config("configs/default.yaml")
df = generate_synthetic(
    cfg["num_users"],
    cfg["num_items"],
    cfg["num_categories"],
    cfg["min_interactions"],
    cfg["max_interactions"],
    cfg["seed"],
)
data = encode_and_split(df)

model = ASMoE(
    data["num_items"],
    data["num_categories"],
    data["item_to_category"],
    cfg["max_len"],
    cfg["embedding_dim"],
    cfg["num_heads"],
    cfg["num_layers"],
    cfg["num_experts"],
    cfg["top_k"],
    cfg["dropout"],
)

checkpoint_path = Path("outputs_v2/best_asmoe_stage2.pt")
try:
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    st.success(f"Loaded Stage-2 checkpoint: `{checkpoint_path}`")
except Exception as exc:
    st.error(
        f"Could not load `{checkpoint_path}`. Make sure the checkpoint exists and run "
        "Streamlit from the project root. If the model architecture/config changed, retrain "
        "the checkpoint with the current code."
    )
    st.exception(exc)
    st.stop()

model.eval()


def get_category(item_id: int) -> int:
    """Return the category from the mapping already provided by encode_and_split."""
    mapping = data["item_to_category"]
    try:
        return int(mapping[item_id])
    except (IndexError, KeyError, TypeError):
        return -1


def get_item_label(item_id: int) -> str:
    # This project uses synthetic interactions and does not supply raw product names.
    # Do not expect a nonexistent data["item_idx_to_raw"] mapping.
    category_id = get_category(item_id)
    if category_id >= 0:
        return f"Demo item {item_id:03d} · Category {category_id}"
    return f"Demo item {item_id:03d}"


user_ids = sorted(data["train"].keys())
uid = st.selectbox(
    "Choose a user",
    user_ids,
    format_func=lambda user_id: f"User {user_id}",
)

history = data["train"][uid][-cfg["max_len"] :]
if not history:
    st.warning("This user has no training interactions. Select another user.")
    st.stop()

history_item_ids = [int(pair[0]) for pair in history]
history_category_ids = [int(pair[1]) for pair in history]

history_left, history_right = st.columns([2, 1])
with history_left:
    st.subheader("Recent interaction history")
    history_df = pd.DataFrame(
        [
            {
                "Step": index + 1,
                "Item": get_item_label(item_id),
                "Item ID": item_id,
                "Category": f"Category {category_id}",
            }
            for index, (item_id, category_id) in enumerate(history)
        ]
    )
    st.dataframe(history_df, use_container_width=True, hide_index=True)

with history_right:
    st.subheader("History summary")
    st.metric("Interactions shown", len(history))
    st.metric("Distinct items", len(set(history_item_ids)))
    st.metric("Distinct categories", len(set(history_category_ids)))

items = torch.tensor([history_item_ids], dtype=torch.long)
categories = torch.tensor([history_category_ids], dtype=torch.long)
valid_mask = torch.ones_like(items, dtype=torch.bool)

with torch.no_grad():
    scores, _ = model(items, categories, valid_mask)
    scores = scores[0].clone()

# Filter every item the user has already interacted with.
seen_ids = sorted(
    {item_id for item_id in history_item_ids if 0 <= item_id < scores.numel()}
)
if seen_ids:
    scores[seen_ids] = -float("inf")

num_candidates = int(torch.isfinite(scores).sum().item())
if num_candidates == 0:
    st.warning("No unseen candidate items are available.")
    st.stop()

top_k = min(10, num_candidates)
top_scores, top_ids = torch.topk(scores, top_k)
recommendations = []
for rank, (item_tensor, score_tensor) in enumerate(
    zip(top_ids, top_scores), start=1
):
    item_id = int(item_tensor.item())
    category_id = get_category(item_id)
    recommendations.append(
        {
            "Rank": rank,
            "Item": get_item_label(item_id),
            "Item ID": item_id,
            "Category": f"Category {category_id}" if category_id >= 0 else "Unknown",
            "Category ID": category_id,
            "Ranking score": round(float(score_tensor.item()), 4),
        }
    )

st.subheader("Top recommendations")
st.caption("Previously seen items are excluded. Ranking scores are not probabilities.")
recommendations_df = pd.DataFrame(recommendations)
st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
st.download_button(
    "Download recommendations as CSV",
    data=recommendations_df.to_csv(index=False).encode("utf-8"),
    file_name=f"asmoe_recommendations_user_{uid}.csv",
    mime="text/csv",
)

st.subheader("Expert routing")
router = getattr(model, "_last_router", None)
if router and "probs" in router:
    probs = router["probs"][0].detach().cpu().flatten()
    expert_df = pd.DataFrame(
        {
            "Expert": [f"Expert {i}" for i in range(len(probs))],
            "Probability": [float(prob) for prob in probs.tolist()],
        }
    ).sort_values("Probability", ascending=False)
    st.bar_chart(expert_df.set_index("Expert")["Probability"])

    if "top_idx" in router:
        selected_experts = router["top_idx"][0].detach().cpu().flatten().tolist()
    else:
        selected_experts = torch.topk(
            probs, min(int(cfg["top_k"]), len(probs))
        ).indices.tolist()

    st.write("Selected experts")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Expert": f"Expert {index}",
                    "Router probability": round(float(probs[index].item()), 3),
                }
                for index in selected_experts
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("The model did not expose router probabilities for this prediction.")

st.subheader("Model evaluation comparison")
comparison_path = Path("outputs_v2/comparison.csv")
if comparison_path.exists():
    try:
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        numeric_columns = [
            column
            for column in comparison_df.columns
            if pd.api.types.is_numeric_dtype(comparison_df[column])
        ]
        name_columns = [
            column
            for column in comparison_df.columns
            if column.lower() in {"model", "name", "method", "approach"}
        ]
        if numeric_columns:
            chart_df = comparison_df.copy()
            if name_columns:
                chart_df = chart_df.set_index(name_columns[0])
            st.bar_chart(chart_df[numeric_columns])
    except Exception as exc:
        st.warning(f"Could not read `{comparison_path}`: {exc}")
else:
    st.info(
        f"`{comparison_path}` was not found. Run the evaluation pipeline to show saved metrics."
    )

with st.expander("Experiment settings and interpretation"):
    st.markdown(
        f"""
        - **Dataset:** Synthetic sequential interactions
        - **Model:** ASMoE Stage 2
        - **Checkpoint:** `{checkpoint_path}`
        - **Sequence length:** Up to {cfg["max_len"]} recent interactions
        - **Item labels:** Generated demo labels, because this synthetic dataset does not provide real product names
        - **Ranking score:** Used to order recommendations; not a probability
        - **Limitation:** Metrics demonstrate this synthetic experiment only and should not be interpreted as real-world performance
        """
    )
