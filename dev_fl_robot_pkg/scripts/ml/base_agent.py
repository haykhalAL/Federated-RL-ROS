from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass