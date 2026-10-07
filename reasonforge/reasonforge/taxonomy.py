from __future__ import annotations

import json
import os
from .models import Taxonomy, Factor


DEFAULT_TAXONOMY = Taxonomy(
    domain="customer support intent classification",
    task="Generate realistic customer support requests with an intent label",
    factors=[
        Factor(
            name="intent",
            values=[
                "refund_request",
                "payment_failed",
                "account_locked",
                "shipping_delay",
                "wrong_item",
                "subscription_cancel",
            ],
        ),
        Factor(
            name="urgency",
            values=["low", "normal", "high", "critical"],
        ),
        Factor(
            name="channel",
            values=["mobile_app", "web", "email", "phone"],
        ),
        Factor(
            name="customer_context",
            values=[
                "first_time_customer",
                "repeat_customer",
                "business_customer",
            ],
        ),
        Factor(
            name="complexity",
            values=["easy", "medium", "hard"],
        ),
    ],
)


def build_taxonomy(domain: str, task: str, provider: str = "local") -> Taxonomy:
    """
    Returns a controllable semantic taxonomy.

    The local implementation is intentionally deterministic so the repository
    can run without an external model. An OpenAI provider can be added later
    without changing the rest of the pipeline.
    """
    if provider == "local":
        t = DEFAULT_TAXONOMY.model_copy(deep=True)
        t.domain = domain
        t.task = task
        return t

    # Keep external generation isolated. This prevents API-specific logic from
    # leaking into the rest of the pipeline.
    from .generator import OpenAITaxonomyGenerator
    return OpenAITaxonomyGenerator().generate_taxonomy(domain, task)
