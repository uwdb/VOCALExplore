from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

class Priority(IntEnum):
    USER = 0
    BACKGROUND = 1
    DEFAULT = 1
    CHORE = 2

class UserPriority:
    priority: Priority = Priority.DEFAULT

class ChorePriority:
    priority: Priority = Priority.DEFAULT

@dataclass(order=True)
class PriorityTask:
    priority: int
    task_id: int
    item: Any=field(compare=False)
    name: str=field(compare=False)
    prepare: Any=field(compare=False, default_factory=(lambda: None))

class AbstractScheduler:
    def context(self):
        raise NotImplementedError

    def schedule(self, name, fn, callback=None, priority: Priority = Priority.DEFAULT, prepare=None):
        raise NotImplementedError

    def tasks_are_waiting(self) -> bool:
        raise NotImplementedError

    def suspend_lowp(self):
        raise NotImplementedError

    def resume_lowp(self):
        raise NotImplementedError

    def suspend_all(self):
        raise NotImplementedError

    def resume_all(self):
        raise NotImplementedError
