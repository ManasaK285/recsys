"""Streamlit dashboard for the Strategic Bargaining Lab.

This is a synthetic, research-inspired simulator, not a reproduction of the
ACM paper's exact protocol. All policy archetypes are hand-coded.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from bargaining_lab.simulation import (
    STRATEGIES,
    run_experiment,
    simulate_game,
    summarize_pairwise,
)

STRATEGY_LABELS = {
    "llm_conservative": "LLM-conservative",
    "human_fairness": "Human-fairness",
    "bayesian_aggressive": "Bayesian-aggressive",
}

st.set_page_config(page_title="Strategic Bargaining Lab", page_icon="🤝", layout="wide")
st.title("🤝 Strategic Tradeoffs in Multi-Agent Bargaining")
st.caption(
    "Research-inspired sandbox comparing conservative, fairness-oriented, "
    "and aggressive policy archetypes."
)

st.info(
    "**Research prototype:** This is a synthetic policy simulation. It does not call real LLMs, "
    "contain human-subject data, or reproduce the ACM paper's exact game protocol."
)

with st.sidebar:
    st.header("Main experiment controls")
    games = st.slider("Number of games", 50, 2000, 500, step=50)
    n_agents = st.slider("Agents per game", 2, 10, 4)
    max_rounds = st.slider("Maximum rounds", 1, 30, 12)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7)
    run = st.button("Run experiment", type="primary")

if "main_result" not in st.session_state:
    st.session_state.main_result = None

if run or st.session_state.main_result is None:
    rounds, agents, summary = run_experiment(
        n_games=games,
        n_agents=n_agents,
        max_rounds=max_rounds,
        seed=int(seed),
    )
    st.session_state.main_result = (rounds, agents, summary)

rounds, agents, summary = st.session_state.main_result
pairwise = summarize_pairwise(rounds)
accepted = float(rounds["offer_accepted"].mean()) if not rounds.empty else 0.0
agreement = (
    float(agents.groupby("game_id")["game_agreement"].max().mean())
    if not agents.empty else 0.0
)
mean_rounds = (
    float(agents.groupby("game_id")["rounds_played"].first().mean())
    if not agents.empty else 0.0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games simulated", f"{agents.game_id.nunique():,}")
c2.metric("Game agreement rate", f"{agreement:.1%}")
c3.metric("Offer acceptance rate", f"{accepted:.1%}")
c4.metric("Mean rounds per game", f"{mean_rounds:.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Strategy outcomes", "Offers & acceptance", "Inspect a game", "Strategy-mix study", "Data"]
)

with tab1:
    st.subheader("Accumulated payoff by strategy")
    if not summary.empty:
        display_summary = summary.copy()
        display_summary["strategy"] = display_summary["strategy"].map(STRATEGY_LABELS)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(display_summary["strategy"], display_summary["mean_payoff"])
        ax.set_ylabel("Mean payoff per agent")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(display_summary, use_container_width=True)
    st.caption(
        "Interpret payoff alongside agreement and rejection rates; a strategy's value depends "
        "on the opponent mix and game rules."
    )

with tab2:
    st.subheader("Proposal behavior by proposer archetype")
    if not pairwise.empty:
        display_pairwise = pairwise.copy()
        display_pairwise["proposer_strategy"] = display_pairwise["proposer_strategy"].map(STRATEGY_LABELS)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(display_pairwise["proposer_strategy"], display_pairwise["mean_offer"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean share offered to responder")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(display_pairwise["proposer_strategy"], display_pairwise["acceptance_rate"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Offer acceptance rate")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.dataframe(display_pairwise, use_container_width=True)
    else:
        st.warning("No offer records were generated for this run.")

with tab3:
    st.subheader("Replay one deterministic game")
    game_seed = st.number_input("Game seed", min_value=0, max_value=999999, value=23, key="game_seed")
    if st.button("Replay game"):
        replay_rounds, replay_agents = simulate_game(
            game_id=0,
            n_agents=n_agents,
            max_rounds=max_rounds,
            seed=int(game_seed),
        )
        st.markdown("**Round-by-round offers and responses**")
        st.dataframe(pd.DataFrame([record.__dict__ for record in replay_rounds]), use_container_width=True)
        st.markdown("**Agent outcomes**")
        st.dataframe(pd.DataFrame(replay_agents), use_container_width=True)

with tab4:
    st.subheader("How does the strategy mix change outcomes?")
    st.write(
        "Compare four predefined populations while keeping the game mechanics the same. "
        "Each scenario uses a different probability of assigning each hand-coded strategy to an agent."
    )
    mix_games = st.slider("Games per scenario", 50, 1000, 250, step=50, key="mix_games")
    mix_agents = st.slider("Agents per scenario game", 2, 10, n_agents, key="mix_agents")
    mix_rounds = st.slider("Round limit for mix study", 1, 30, max_rounds, key="mix_rounds")
    mix_seed = st.number_input("Mix-study seed", min_value=0, max_value=999999, value=int(seed) + 100, key="mix_seed")

    scenarios = {
        "Balanced": {
            "llm_conservative": 1,
            "human_fairness": 1,
            "bayesian_aggressive": 1,
        },
        "Conservative-heavy": {
            "llm_conservative": 4,
            "human_fairness": 1,
            "bayesian_aggressive": 1,
        },
        "Fairness-heavy": {
            "llm_conservative": 1,
            "human_fairness": 4,
            "bayesian_aggressive": 1,
        },
        "Aggressive-heavy": {
            "llm_conservative": 1,
            "human_fairness": 1,
            "bayesian_aggressive": 4,
        },
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"Scenario": name, **{STRATEGY_LABELS[k]: v for k, v in mix.items()}}
                for name, mix in scenarios.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Run strategy-mix comparison", type="primary", key="run_mix_study"):
        scenario_rows = []
        strategy_rows = []
        progress = st.progress(0, text="Preparing scenarios...")
        for idx, (scenario_name, mix) in enumerate(scenarios.items(), start=1):
            scenario_rounds, scenario_agents, scenario_summary = run_experiment(
                n_games=mix_games,
                n_agents=mix_agents,
                max_rounds=mix_rounds,
                seed=int(mix_seed) + (idx - 1) * 100000,
                mix=mix,
            )
            scenario_agreement = float(
                scenario_agents.groupby("game_id")["game_agreement"].max().mean()
            ) if not scenario_agents.empty else 0.0
            scenario_acceptance = float(scenario_rounds["offer_accepted"].mean()) if not scenario_rounds.empty else 0.0
            scenario_mean_rounds = float(
                scenario_agents.groupby("game_id")["rounds_played"].first().mean()
            ) if not scenario_agents.empty else 0.0
            scenario_rows.append({
                "scenario": scenario_name,
                "games": int(scenario_agents["game_id"].nunique()),
                "agreement_rate": scenario_agreement,
                "offer_acceptance_rate": scenario_acceptance,
                "mean_rounds": scenario_mean_rounds,
            })
            for row in scenario_summary.to_dict(orient="records"):
                strategy_rows.append({"scenario": scenario_name, **row})
            progress.progress(idx / len(scenarios), text=f"Completed {scenario_name}")

        mix_overview = pd.DataFrame(scenario_rows)
        mix_strategy = pd.DataFrame(strategy_rows)
        st.session_state.mix_overview = mix_overview
        st.session_state.mix_strategy = mix_strategy
        st.session_state.mix_config = {
            "games": mix_games, "agents": mix_agents, "rounds": mix_rounds, "seed": int(mix_seed)
        }
        progress.empty()

    if "mix_overview" in st.session_state:
        mix_overview = st.session_state.mix_overview
        mix_strategy = st.session_state.mix_strategy
        st.markdown("#### Agreement and acceptance by population")
        display_overview = mix_overview.copy()
        for col in ["agreement_rate", "offer_acceptance_rate"]:
            display_overview[col] = display_overview[col].map(lambda x: f"{x:.1%}")
        display_overview["mean_rounds"] = display_overview["mean_rounds"].map(lambda x: f"{x:.2f}")
        st.dataframe(display_overview, hide_index=True, use_container_width=True)

        st.markdown("#### Mean payoff by strategy and population")
        if not mix_strategy.empty:
            pivot = mix_strategy.pivot(index="scenario", columns="strategy", values="mean_payoff")
            pivot = pivot.rename(columns=STRATEGY_LABELS)
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean payoff per agent")
            ax.set_xlabel("Population scenario")
            ax.tick_params(axis="x", rotation=15)
            ax.legend(title="Strategy")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            display_mix_strategy = mix_strategy.copy()
            display_mix_strategy["strategy"] = display_mix_strategy["strategy"].map(STRATEGY_LABELS)
            st.dataframe(display_mix_strategy, hide_index=True, use_container_width=True)

        st.download_button(
            "Download scenario metrics CSV",
            mix_overview.to_csv(index=False),
            "strategy_mix_overview.csv",
            "text/csv",
        )
        st.download_button(
            "Download strategy-by-scenario CSV",
            mix_strategy.to_csv(index=False),
            "strategy_mix_payoffs.csv",
            "text/csv",
        )
        st.caption(
            "Results depend on the simulator's assumptions and random seed. These are synthetic "
            "comparisons, not empirical estimates about real people or deployed LLMs."
        )
    else:
        st.info("Choose the experiment size above, then click **Run strategy-mix comparison**.")

with tab5:
    st.subheader("Round-level observations")
    st.dataframe(rounds, use_container_width=True, height=350)
    st.download_button("Download rounds CSV", rounds.to_csv(index=False), "rounds.csv", "text/csv")
    st.download_button(
        "Download strategy summary CSV",
        summary.to_csv(index=False),
        "strategy_summary.csv",
        "text/csv",
    )
    st.download_button(
        "Download offer acceptance CSV",
        pairwise.to_csv(index=False),
        "offer_acceptance_summary.csv",
        "text/csv",
    )

st.divider()
st.markdown(
    "**Interpretation checklist**\n"
    "- Do conservative proposals get accepted more often?\n"
    "- Does aggressive proposing improve the proposer’s payoff while increasing rejection risk?\n"
    "- Do fairness-oriented proposals trade off immediate surplus against agreement probability?\n"
    "- Does the ranking change when the strategy mix, number of agents, seed, or deadline changes?"
)
