#!/usr/bin/env python3

from abc import ABC, abstractmethod


class FederatedStrategy(ABC):
    """
    Base interface for federated aggregation strategies.

    Each strategy receives local model updates and produces
    one global model.
    """

    @abstractmethod
    def aggregate(self, updates):
        """
        Aggregate local model updates.

        Args:
            updates: list of dictionaries:
                {
                    "client_id": str,
                    "state_dict": dict,
                    "num_samples": int
                }

        Returns:
            global_state_dict
        """
        raise NotImplementedError