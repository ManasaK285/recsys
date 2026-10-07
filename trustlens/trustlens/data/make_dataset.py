"""
TrustLens synthetic dataset generator.

Generates:
    2400 unique responses
    1200 human
    1200 AI

The generator also records template metadata so that
Experiment 06 can perform proper paired style-template
generalization.

Important:
    Human and AI style templates have DIFFERENT IDs.

    human_style_0
    human_style_1
    ...
    human_style_4

    ai_style_0
    ai_style_1
    ...
    ai_style_4
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "trustlens.csv"
)

N_PER_SOURCE = 1200


# =====================================================================
# SCENARIOS
# =====================================================================

SCENARIOS = [
    {
        "scenario_id": "trolley",
        "scenario_type": "utilitarian",
        "scenario": (
            "A train is heading toward five people. "
            "Would you divert it onto another track where "
            "it would kill one person?"
        ),
        "reasons": [
            "the outcome affects more people",
            "reducing total harm is important",
            "the consequences should be considered",
            "one death may prevent several deaths",
            "the decision involves competing harms",
            "the number of people affected matters",
        ],
    },
    {
        "scenario_id": "bridge",
        "scenario_type": "utilitarian",
        "scenario": (
            "Would you push one person from a bridge to stop "
            "a train from killing five people?"
        ),
        "reasons": [
            "actively causing harm feels different from allowing harm",
            "the person would be directly used as a means",
            "the consequences still involve competing harms",
            "the distinction between action and inaction matters",
            "saving more people does not automatically justify the action",
            "the directness of the harm is important",
        ],
    },
    {
        "scenario_id": "medicine",
        "scenario_type": "allocation",
        "scenario": (
            "A hospital has one scarce treatment and two patients "
            "need it. Should it go to the patient with the higher "
            "chance of recovery?"
        ),
        "reasons": [
            "the treatment may produce a better medical outcome",
            "limited resources require difficult prioritization",
            "fair access should still be considered",
            "the probability of recovery provides relevant evidence",
            "each patient has a claim to consideration",
            "medical benefit is one important allocation factor",
        ],
    },
    {
        "scenario_id": "privacy",
        "scenario_type": "privacy",
        "scenario": (
            "A company can analyze private customer data to improve "
            "safety. Should it do so without explicit permission?"
        ),
        "reasons": [
            "privacy is important even when the intended purpose is beneficial",
            "customers may reasonably expect control over their information",
            "the safety benefit should be weighed against the privacy cost",
            "using information without permission can undermine trust",
            "the purpose of the data use does not remove consent concerns",
            "responsible data practices require meaningful safeguards",
        ],
    },
    {
        "scenario_id": "lying",
        "scenario_type": "honesty",
        "scenario": (
            "Is it acceptable to tell a small lie if it prevents "
            "serious emotional harm to another person?"
        ),
        "reasons": [
            "protecting someone from serious harm can matter",
            "honesty is generally an important principle",
            "the seriousness of the consequences affects the decision",
            "a small lie may sometimes prevent greater harm",
            "trust can be damaged when people discover deception",
            "the intention behind the lie is relevant",
        ],
    },
    {
        "scenario_id": "autonomy",
        "scenario_type": "autonomy",
        "scenario": (
            "Should a competent adult be allowed to reject a medical "
            "treatment even when it would save their life?"
        ),
        "reasons": [
            "competent adults generally have authority over their own bodies",
            "saving a life does not automatically override personal choice",
            "informed consent is an important consideration",
            "the person's values and preferences should matter",
            "medical benefit and individual autonomy can conflict",
            "respect for voluntary decisions is important",
        ],
    },
    {
        "scenario_id": "fairness",
        "scenario_type": "allocation",
        "scenario": (
            "Should a limited scholarship be given to the applicant "
            "with greater financial need rather than the applicant "
            "with higher grades?"
        ),
        "reasons": [
            "financial need can be an important fairness consideration",
            "academic achievement is also relevant",
            "equal opportunity may require considering different circumstances",
            "the purpose of the scholarship affects the decision",
            "supporting disadvantaged applicants can promote access",
            "merit and need represent competing allocation principles",
        ],
    },
]


# =====================================================================
# SHARED LANGUAGE BANKS
# =====================================================================

OPENINGS = [
    "I think",
    "In my view",
    "My initial reaction is that",
    "I would say that",
    "It seems to me that",
    "I would lean toward the view that",
    "My perspective is that",
    "I tend to think that",
    "I would argue that",
    "One way to look at this is that",
    "The way I see it is that",
    "I would consider this",
    "My first thought is that",
    "I would approach this by considering whether",
    "I would be inclined to say that",
]


TRANSITIONS = [
    "At the same time,",
    "However,",
    "That said,",
    "On the other hand,",
    "Still,",
    "Even so,",
    "There is also a case for saying that",
    "Another consideration is that",
    "It is also worth noting that",
    "A competing point is that",
    "The other side of the issue is that",
    "I would also consider whether",
]


CAVEATS = [
    "The context could change how I view the situation.",
    "The details of the situation would matter.",
    "I would want to know more before being completely certain.",
    "There may be reasonable arguments on both sides.",
    "The consequences should be considered carefully.",
    "The circumstances could make the balance different.",
    "I do not think the answer is completely straightforward.",
    "There is some uncertainty in making that judgment.",
    "The competing concern should not simply be ignored.",
    "The decision involves a genuine tradeoff.",
    "Different priorities could lead to a different conclusion.",
    "The specific facts would affect the final judgment.",
]


ENDINGS = [
    "Overall, that is where I would land.",
    "For those reasons, that seems reasonable to me.",
    "That would be my preference in this situation.",
    "So I would lean in that direction.",
    "That seems like the more appropriate approach.",
    "For me, the balance points that way.",
    "That is the conclusion I would be most comfortable with.",
    "I would therefore favor that approach.",
    "That seems to best account for the relevant concerns.",
    "So that is the position I would take.",
    "On balance, I would approach it that way.",
    "That is how I would resolve the tradeoff.",
]


# =====================================================================
# HUMAN SOURCE-SPECIFIC TEMPLATES
# =====================================================================

HUMAN_FRAMES = [
    "because {reason}",
    "mainly because {reason}",
    "especially because {reason}",
    "largely because {reason}",
    "partly because {reason}",
    "because I think {reason}",
]


HUMAN_PERSONAL = [
    "Personally, I would put some weight on that.",
    "From my perspective, that matters.",
    "I would be hesitant to ignore that.",
    "I think that is worth keeping in mind.",
    "That seems important to me.",
]


# =====================================================================
# AI SOURCE-SPECIFIC TEMPLATES
# =====================================================================

AI_FRAMES = [
    "because {reason}",
    "primarily because {reason}",
    "largely because {reason}",
    "given that {reason}",
    "since {reason}",
    "because the relevant issue is that {reason}",
]


AI_STRUCTURED = [
    "This also suggests that the competing consideration should not be ignored.",
    "The competing consideration should therefore remain part of the analysis.",
    "This makes the tradeoff between the relevant factors important.",
    "The alternative should also be considered before reaching a conclusion.",
    "This indicates that both sides of the decision should be evaluated.",
]

# =====================================================================
# DECISION TEMPLATES
# =====================================================================

YES_PHRASES = [
    "I would say yes.",
    "I would support doing that.",
    "I would be inclined to allow it.",
    "I think that is acceptable.",
    "I would favor that option.",
    "I would generally support it.",
]


NO_PHRASES = [
    "I would say no.",
    "I would not support doing that.",
    "I would be hesitant to allow it.",
    "I do not think that is acceptable.",
    "I would favor the alternative.",
    "I would generally oppose it.",
]


# =====================================================================
# TEMPLATE ID DICTIONARIES
# =====================================================================

# IMPORTANT:
# Human and AI styles have independent namespaces.

HUMAN_STYLE_TEMPLATES = {
    f"human_style_{i}": phrase
    for i, phrase in enumerate(HUMAN_PERSONAL)
}


AI_STYLE_TEMPLATES = {
    f"ai_style_{i}": phrase
    for i, phrase in enumerate(AI_STRUCTURED)
}


# =====================================================================
# RANDOM SELECTION HELPERS
# =====================================================================

def choose_with_id(items):
    """
    Randomly select an item from a list.

    Returns:
        selected_item
        selected_index
    """

    index = random.randrange(len(items))

    return items[index], index


# =====================================================================
# RESPONSE GENERATOR
# =====================================================================

def generate_response(
    scenario,
    source,
    judgment,
):
    """
    Generate one response and its metadata.
    """

    # -------------------------------------------------------------
    # Shared templates
    # -------------------------------------------------------------

    opening, opening_id = choose_with_id(
        OPENINGS
    )

    transition, transition_id = choose_with_id(
        TRANSITIONS
    )

    caveat, caveat_id = choose_with_id(
        CAVEATS
    )

    ending, ending_id = choose_with_id(
        ENDINGS
    )

    # -------------------------------------------------------------
    # Source-specific frame and style
    # -------------------------------------------------------------

    if source == "human":

        frame_template, frame_id = choose_with_id(
            HUMAN_FRAMES
        )

        style_id = random.choice(
            list(HUMAN_STYLE_TEMPLATES.keys())
        )

        style_phrase = HUMAN_STYLE_TEMPLATES[
            style_id
        ]

        style_family = "personal"

    else:

        frame_template, frame_id = choose_with_id(
            AI_FRAMES
        )

        style_id = random.choice(
            list(AI_STYLE_TEMPLATES.keys())
        )

        style_phrase = AI_STYLE_TEMPLATES[
            style_id
        ]

        style_family = "structured"

    # -------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------

    if judgment == 1:

        decision_phrase, decision_id = choose_with_id(
            YES_PHRASES
        )

    else:

        decision_phrase, decision_id = choose_with_id(
            NO_PHRASES
        )

    # -------------------------------------------------------------
    # Reason
    # -------------------------------------------------------------

    reason = random.choice(
        scenario["reasons"]
    )

    frame = frame_template.format(
        reason=reason
    )

    # -------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------

    response = (
        f"{opening} "
        f"{decision_phrase} "
        f"{frame}. "
        f"{transition} "
        f"{style_phrase} "
        f"{caveat} "
        f"{ending}"
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {
        "opening_id": opening_id,
        "transition_id": transition_id,
        "caveat_id": caveat_id,
        "ending_id": ending_id,
        "frame_id": frame_id,
        "style_id": style_id,
        "style_family": style_family,
        "decision_id": decision_id,
    }

    return response, metadata


# =====================================================================
# DATASET GENERATION
# =====================================================================

rows = []

used_responses = set()

human_count = 0
ai_count = 0


while len(rows) < N_PER_SOURCE * 2:

    # -------------------------------------------------------------
    # Select source while maintaining exact balance
    # -------------------------------------------------------------

    if human_count >= N_PER_SOURCE:

        source = "ai"

    elif ai_count >= N_PER_SOURCE:

        source = "human"

    else:

        source = random.choice(
            ["human", "ai"]
        )

    # -------------------------------------------------------------
    # Scenario
    # -------------------------------------------------------------

    scenario = random.choice(
        SCENARIOS
    )

    # -------------------------------------------------------------
    # Judgment
    # -------------------------------------------------------------

    judgment = random.randint(
        0,
        1,
    )

    # -------------------------------------------------------------
    # Generate response
    # -------------------------------------------------------------

    response, metadata = generate_response(
        scenario=scenario,
        source=source,
        judgment=judgment,
    )

    # -------------------------------------------------------------
    # Guarantee unique response text
    # -------------------------------------------------------------

    if response in used_responses:
        continue

    used_responses.add(
        response
    )

    # -------------------------------------------------------------
    # Perceived AI
    # -------------------------------------------------------------

    if source == "ai":

        perceived_ai = np.random.binomial(
            1,
            0.65,
        )

    else:

        perceived_ai = np.random.binomial(
            1,
            0.35,
        )

    # -------------------------------------------------------------
    # Agreement
    # -------------------------------------------------------------

    if judgment == 1:

        agreement_probability = 0.62

    else:

        agreement_probability = 0.38

    agreement_probability -= (
        0.03 * perceived_ai
    )

    agreement_probability += np.random.uniform(
        -0.05,
        0.05,
    )

    agreement_probability = np.clip(
        agreement_probability,
        0.05,
        0.95,
    )

    agreement = np.random.binomial(
        1,
        agreement_probability,
    )

    # -------------------------------------------------------------
    # Append row
    # -------------------------------------------------------------

    rows.append(
        {
            "response_id": len(rows),

            "scenario_id": scenario[
                "scenario_id"
            ],

            "scenario": scenario[
                "scenario"
            ],

            "scenario_type": scenario[
                "scenario_type"
            ],

            "response": response,

            "source": source,

            "judgment": judgment,

            "perceived_ai": perceived_ai,

            "agreement": agreement,

            # -----------------------------------------------------
            # Template metadata
            # -----------------------------------------------------

            "opening_id": metadata[
                "opening_id"
            ],

            "transition_id": metadata[
                "transition_id"
            ],

            "caveat_id": metadata[
                "caveat_id"
            ],

            "ending_id": metadata[
                "ending_id"
            ],

            "frame_id": metadata[
                "frame_id"
            ],

            "style_id": metadata[
                "style_id"
            ],

            "style_family": metadata[
                "style_family"
            ],

            "decision_id": metadata[
                "decision_id"
            ],
        }
    )

    # -------------------------------------------------------------
    # Update source counters
    # -------------------------------------------------------------

    if source == "human":

        human_count += 1

    else:

        ai_count += 1


# =====================================================================
# DATAFRAME
# =====================================================================

df = pd.DataFrame(
    rows
)


# =====================================================================
# SHUFFLE
# =====================================================================

df = df.sample(
    frac=1,
    random_state=SEED,
).reset_index(
    drop=True
)


# Reassign IDs after shuffle.

df["response_id"] = np.arange(
    len(df)
)


# =====================================================================
# VALIDATION
# =====================================================================

assert len(df) == 2400

assert (
    df["response"].nunique()
    == 2400
)

assert (
    df["source"]
    .value_counts()["human"]
    == 1200
)

assert (
    df["source"]
    .value_counts()["ai"]
    == 1200
)

assert (
    df["response"].isna().sum()
    == 0
)

assert (
    df["style_family"]
    .isin(
        [
            "personal",
            "structured",
        ]
    )
    .all()
)

# Human must only use human_style_*.

assert (
    df.loc[
        df["source"] == "human",
        "style_id",
    ]
    .str.startswith("human_style_")
    .all()
)

# AI must only use ai_style_*.

assert (
    df.loc[
        df["source"] == "ai",
        "style_id",
    ]
    .str.startswith("ai_style_")
    .all()
)


# =====================================================================
# SAVE
# =====================================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =====================================================================
# SUMMARY
# =====================================================================

print(
    f"saved {len(df)} rows to {OUTPUT_PATH}"
)

print(
    "\nUnique responses:",
    df["response"].nunique(),
)

print(
    "\nSource distribution:"
)

print(
    df["source"].value_counts()
)

print(
    "\nScenario distribution:"
)

print(
    df["scenario_id"].value_counts()
)

print(
    "\nPerceived AI distribution:"
)

print(
    df["perceived_ai"].value_counts()
)

print(
    "\nAgreement distribution:"
)

print(
    df["agreement"].value_counts()
)

print(
    "\nStyle family distribution:"
)

print(
    df["style_family"].value_counts()
)

print(
    "\nStyle templates:"
)

print(
    df["style_id"].value_counts()
)

print(
    "\nStyle templates by source:"
)

print(
    df.groupby("source")["style_id"]
    .nunique()
)

print(
    "\nTemplate metadata:"
)

print(
    "Openings:",
    df["opening_id"].nunique()
)

print(
    "Transitions:",
    df["transition_id"].nunique()
)

print(
    "Caveats:",
    df["caveat_id"].nunique()
)

print(
    "Endings:",
    df["ending_id"].nunique()
)

print(
    "Frames:",
    df["frame_id"].nunique()
)

print(
    "Style templates:",
    df["style_id"].nunique()
)

print(
    "Decision phrases:",
    df["decision_id"].nunique()
)