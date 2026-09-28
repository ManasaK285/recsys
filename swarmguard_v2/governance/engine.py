from environment.models import Event, EventType


class Governance:
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config

    def process(
        self,
        swarm,
        agent_id,
        solution,
        flagged,
        exploited,
        attempted_shortcut,
        round_idx,
    ):
        if not self.policy.audit or not flagged:
            return

        swarm.log(
            Event(
                round_idx,
                EventType.VIOLATION_DETECTED,
                agent_id,
                solution.task_id,
                {
                    "exploited": exploited,
                    "attempted_shortcut": attempted_shortcut,
                },
            )
        )

        if exploited:
            if self.policy.whistleblowing:
                swarm.kb.publish(
                    agent_id,
                    "ALERT",
                    f"Potential evaluator-gaming by {agent_id}",
                    round_idx,
                )

                swarm.bus.broadcast(
                    agent_id,
                    "ALERT",
                    f"Potential exploit use by {agent_id}",
                    round_idx,
                )

                swarm.log(
                    Event(
                        round_idx,
                        EventType.WHISTLEBLOWN,
                        agent_id,
                        solution.task_id,
                        {},
                    )
                )

                swarm.log(
                    Event(
                        round_idx,
                        EventType.ALERT_SENT,
                        agent_id,
                        solution.task_id,
                        {},
                    )
                )

            if self.policy.quarantine:
                agent = swarm.agents[agent_id]

                if (
                    agent.state.quarantined_until is None
                    or round_idx >= agent.state.quarantined_until
                ):
                    agent.state.quarantined_until = (
                        round_idx + self.config.quarantine_rounds
                    )

                    swarm.log(
                        Event(
                            round_idx,
                            EventType.AGENT_QUARANTINED,
                            agent_id,
                            solution.task_id,
                            {
                                "until": agent.state.quarantined_until
                            },
                        )
                    )

                    swarm.log(
                        Event(
                            round_idx,
                            EventType.TASK_REOPENED,
                            agent_id,
                            solution.task_id,
                            {},
                        )
                    )

        elif attempted_shortcut:
            # The auditor correctly identified a shortcut attempt,
            # but the strict evaluator rejected it. This is not a
            # false accusation.
            swarm.log(
                Event(
                    round_idx,
                    EventType.VIOLATION_DETECTED,
                    agent_id,
                    solution.task_id,
                    {
                        "exploited": False,
                        "attempted_shortcut": True,
                        "blocked": True,
                    },
                )
            )

        else:
            # Legitimate solution incorrectly flagged by the auditor.
            swarm.log(
                Event(
                    round_idx,
                    EventType.FALSE_ACCUSATION,
                    agent_id,
                    solution.task_id,
                    {},
                )
            )

    def recover(self, swarm, round_idx):
        if not self.policy.recovery:
            return

        for agent in swarm.agents.values():
            until = agent.state.quarantined_until

            if until is None or round_idx < until:
                continue

            if self.config.recovery_probability <= 0:
                continue

            if swarm.rng.random() > self.config.recovery_probability:
                continue

            agent.state.quarantined_until = None

            swarm.log(
                Event(
                    round_idx,
                    EventType.AGENT_RECOVERED,
                    agent.state.agent_id,
                    None,
                    {},
                )
            )
