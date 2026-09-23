#!/usr/bin/env python3

from .fedavg import FedAvgStrategy
from .checkpoint import FederatedCheckpointManager


class FederatedCoordinator:
    """
    Controls federated rounds.

    Responsibilities:
        - track local progress
        - collect client updates
        - determine when all clients are ready
        - invoke aggregation strategy
        - broadcast global model
        - save checkpoints
    """

    def __init__(
        self,
        strategy,
        local_episodes,
        communication_rounds,
        model_dir
    ):

        self.strategy = strategy

        self.local_episodes = int(
            local_episodes
        )

        self.communication_rounds = int(
            communication_rounds
        )

        self.checkpoint_manager = (
            FederatedCheckpointManager(
                model_dir
            )
        )

        self.current_round = 0

        self.updates = {}

    # ========================================================
    # ROUND STATE
    # ========================================================

    def is_round_boundary(
        self,
        completed_episodes
    ):

        return (
            completed_episodes > 0
            and
            completed_episodes
            % self.local_episodes == 0
        )

    def expected_round(
        self,
        completed_episodes
    ):

        return (
            completed_episodes
            // self.local_episodes
        )

    def is_finished(self):

        return (
            self.current_round
            >= self.communication_rounds
        )

    # ========================================================
    # CLIENT UPDATE
    # ========================================================

    def submit_update(
        self,
        client_id,
        agent,
        num_samples
    ):

        self.updates[client_id] = {
            "client_id": client_id,
            "state_dict": (
                agent.get_model_state_dict()
            ),
            "num_samples": int(
                num_samples
            )
        }

    # ========================================================
    # READY CHECK
    # ========================================================

    def all_clients_ready(
        self,
        client_ids
    ):

        return all(
            client_id in self.updates
            for client_id in client_ids
        )

    # ========================================================
    # AGGREGATE
    # ========================================================

    def aggregate(
        self,
        agents
    ):

        if not self.updates:
            raise RuntimeError(
                "No federated updates available."
            )

        self.current_round += 1

        global_state = (
            self.strategy.aggregate(
                list(
                    self.updates.values()
                )
            )
        )

        # ----------------------------------------------------
        # Broadcast global model
        # ----------------------------------------------------

        for client_id, agent in agents.items():

            agent.set_model_state_dict(
                global_state,
                reset_optimizer=True
            )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        sample_counts = {
            client_id:
                update["num_samples"]
            for client_id, update
            in self.updates.items()
        }

        checkpoint_path = (
            self.checkpoint_manager.save(
                global_state,
                self.current_round,
                metadata={
                    "sample_counts":
                        sample_counts,
                    "clients":
                        list(
                            self.updates.keys()
                        )
                }
            )
        )

        # ----------------------------------------------------
        # Clear current round
        # ----------------------------------------------------

        self.updates.clear()

        return {
            "round": self.current_round,
            "state_dict": global_state,
            "sample_counts": sample_counts,
            "checkpoint": checkpoint_path
        }