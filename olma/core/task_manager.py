"""작업 상태 머신: IDLE -> PLANNING -> EXECUTING -> DONE / FAILED."""
from enum import Enum


class State(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    FAILED = "FAILED"


class TaskManager:
    def __init__(self):
        self.state = State.IDLE

    def start_planning(self):
        self.state = State.PLANNING

    def start_executing(self):
        self.state = State.EXECUTING

    def complete(self):
        self.state = State.DONE

    def fail(self):
        self.state = State.FAILED

    def reset(self):
        self.state = State.IDLE
