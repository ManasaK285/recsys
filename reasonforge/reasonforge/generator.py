from __future__ import annotations

import json
import os
import re
from typing import Dict

from .models import Scenario, Taxonomy


INTENT_TEXT = {
    "refund_request": [
        "I want my money back for an order that I no longer need.",
        "How can I request a refund for this purchase?",
        "I was charged for an item I returned and need the payment reversed.",
    ],
    "payment_failed": [
        "My payment keeps getting declined even though the card works elsewhere.",
        "The checkout page will not accept my payment.",
        "I tried to pay twice and both transactions failed.",
    ],
    "account_locked": [
        "I cannot access my account because it is locked.",
        "My login is blocked after several attempts.",
        "The app says my account has been temporarily locked.",
    ],
    "shipping_delay": [
        "My package is later than the promised delivery date.",
        "The tracking page has not moved and my delivery is overdue.",
        "Can you tell me why my shipment is delayed?",
    ],
    "wrong_item": [
        "The package arrived but it contains a different item.",
        "I received the wrong product and need a replacement.",
        "The item in my delivery does not match what I ordered.",
    ],
    "subscription_cancel": [
        "I want to stop my subscription before the next renewal.",
        "Please cancel my recurring membership.",
        "I no longer want the subscription and need it cancelled.",
    ],
}

URGENCY_PHRASES = {
    "low": "There is no immediate deadline.",
    "normal": "I would appreciate help when convenient.",
    "high": "I need this resolved soon.",
    "critical": "This is blocking an important activity and needs urgent attention.",
}

CHANNEL_PHRASES = {
    "mobile_app": "The issue is being reported from the mobile app.",
    "web": "The issue occurred on the website.",
    "email": "The customer is contacting support by email.",
    "phone": "The customer is describing the issue during a phone interaction.",
}

CUSTOMER_PHRASES = {
    "first_time_customer": "The customer is using the service for the first time.",
    "repeat_customer": "The customer has used the service repeatedly.",
    "business_customer": "The customer is acting on behalf of a business.",
}


class LocalGenerator:
    def generate(
        self,
        sample_id: str,
        factors: Dict[str, str],
    ) -> Scenario:
        intent = factors["intent"]
        complexity = factors["complexity"]

        base = INTENT_TEXT[intent][
            (hash(sample_id) % len(INTENT_TEXT[intent]))
        ]

        pieces = [
            base,
            URGENCY_PHRASES[factors["urgency"]],
            CHANNEL_PHRASES[factors["channel"]],
            CUSTOMER_PHRASES[factors["customer_context"]],
        ]

        if complexity == "medium":
            pieces.append(
                "The customer has already tried the standard self-service option."
            )
        elif complexity == "hard":
            pieces.extend(
                [
                    "The customer has already tried the standard self-service option.",
                    "There is an additional constraint that makes a generic response insufficient.",
                    "The response should identify the correct support path rather than guessing.",
                ]
            )

        instruction = " ".join(pieces)

        answer = self._answer(intent, complexity)

        return Scenario(
            sample_id=sample_id,
            intent=intent,
            factors=factors,
            complexity=complexity,
            instruction=instruction,
            expected_label=intent,
            answer=answer,
        )

    @staticmethod
    def _answer(intent: str, complexity: str) -> str:
        actions = {
            "refund_request": "Verify the order and refund eligibility, then initiate or explain the refund process.",
            "payment_failed": "Verify the payment method, identify the decline reason, and provide the appropriate retry or support path.",
            "account_locked": "Verify account ownership and follow the account recovery or unlock procedure.",
            "shipping_delay": "Check tracking and delivery status, then provide the appropriate escalation or updated delivery guidance.",
            "wrong_item": "Verify the order and item received, then initiate replacement or return support.",
            "subscription_cancel": "Verify the subscription and cancellation terms, then cancel or explain the next available cancellation step.",
        }
        return actions[intent]


class OpenAIGenerator:
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6")

    def generate(self, sample_id: str, factors: Dict[str, str]) -> Scenario:
        system = """You generate synthetic customer-support classification examples.
Return ONLY valid JSON with keys:
instruction, expected_label, answer.
Do not copy any real customer. Make the example realistic, diverse, and internally
consistent. The label must exactly match the supplied intent."""
        prompt = {
            "intent": factors["intent"],
            "urgency": factors["urgency"],
            "channel": factors["channel"],
            "customer_context": factors["customer_context"],
            "complexity": factors["complexity"],
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.9,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )

        data = json.loads(response.choices[0].message.content)
        return Scenario(
            sample_id=sample_id,
            intent=factors["intent"],
            factors=factors,
            complexity=factors["complexity"],
            instruction=data["instruction"],
            expected_label=data["expected_label"],
            answer=data["answer"],
        )


class OpenAITaxonomyGenerator:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6")

    def generate_taxonomy(self, domain: str, task: str) -> Taxonomy:
        from .models import Factor, Taxonomy

        prompt = f"""
Design a compact taxonomy for this synthetic-data task.

Domain: {domain}
Task: {task}

Return JSON:
{{
  "factors": [
    {{"name": "...", "values": ["...", "..."]}}
  ]
}}

Include factors that control semantic coverage, context, difficulty, and variation.
Keep it practical: 4-6 factors and 3-6 values per factor.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a dataset mechanism designer."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return Taxonomy(
            domain=domain,
            task=task,
            factors=[Factor(**x) for x in data["factors"]],
        )
