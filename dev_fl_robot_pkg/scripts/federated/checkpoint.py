#!/usr/bin/env python3

import os
import torch


class FederatedCheckpointManager:
    """
    Handles saving global federated models.
    """

    def __init__(self, model_dir):

        self.model_dir = model_dir

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

    def save(
        self,
        state_dict,
        round_number,
        metadata=None
    ):

        filename = (
            f"federated_round_"
            f"{round_number:03d}.pt"
        )

        path = os.path.join(
            self.model_dir,
            filename
        )

        checkpoint = {
            "round": round_number,
            "model_state_dict": state_dict,
            "metadata": metadata or {}
        }

        torch.save(
            checkpoint,
            path
        )

        return path