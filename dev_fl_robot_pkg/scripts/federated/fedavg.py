#!/usr/bin/env python3

import torch

from .strategy import FederatedStrategy


class FedAvgStrategy(FederatedStrategy):
    """
    Sample-weighted Federated Averaging.

    Global model:

        theta_global =
            sum(n_k * theta_k) / sum(n_k)

    where:
        n_k = number of local samples from client k
    """

    def aggregate(self, updates):

        if not updates:
            raise ValueError(
                "FedAvg received no client updates."
            )

        total_samples = sum(
            update["num_samples"]
            for update in updates
        )

        # ----------------------------------------------------
        # Equal weighting fallback
        # ----------------------------------------------------

        if total_samples <= 0:

            client_weight = (
                1.0 / len(updates)
            )

            weights = {
                update["client_id"]:
                    client_weight
                for update in updates
            }

        else:

            weights = {
                update["client_id"]:
                    update["num_samples"]
                    / float(total_samples)
                for update in updates
            }

        # ----------------------------------------------------
        # Use first client to establish parameter structure
        # ----------------------------------------------------

        first_state = updates[0]["state_dict"]

        global_state = {}

        # ----------------------------------------------------
        # Aggregate each model parameter
        # ----------------------------------------------------

        for key in first_state:

            first_tensor = first_state[key]

            # Floating tensors can be averaged normally.
            if torch.is_floating_point(first_tensor):

                aggregated = torch.zeros_like(
                    first_tensor,
                    dtype=torch.float32
                )

                for update in updates:

                    local_tensor = (
                        update["state_dict"][key]
                    )

                    weight = weights[
                        update["client_id"]
                    ]

                    aggregated += (
                        local_tensor.float()
                        * weight
                    )

                global_state[key] = (
                    aggregated.to(
                        dtype=first_tensor.dtype
                    )
                )

            else:
                # Integer / non-floating parameters are copied.
                global_state[key] = first_tensor.clone()

        return global_state