import sys
from enum import Enum, auto
from collections import defaultdict
import typing


class EventType(Enum):
    EXPLORATION_STARTED = auto()
    EXPLORATION_COMPLETED = auto()
    RESCUE_STARTED = auto()
    RESCUE_COMPLETED = auto()

instance = None
can_instantiate = False

class SingletonError(Exception):
    def __init__(self, message):
        super().__init__(message)

class EventManager:
    @staticmethod
    def get_instance():
        global instance
        if instance is None:
            global can_instantiate
            can_instantiate = True
            instance = EventManager()
            can_instantiate = False
        return instance

    def __init__(self):
        global instance
        if not can_instantiate:
            raise SingletonError("Invalid use of singleton class.")
        self.event_connections = defaultdict(list)

    # noinspection PyTypeHints
    def register_callback(self, event: EventType, callback: callable):
        self.event_connections[event].append(callback)

    def unregister_callback(self, event, callback):
        callbacks = self.event_connections.get(event)
        if callbacks:
            try:
                callbacks.remove(callback)
            except ValueError as e:
                print(f"Error: {event.name} is not registered: {e}")

    def emit_event(self, event, *args):
            callbacks = self.event_connections.get(event)
            if callbacks and len(callbacks) > 0:
                # We use a copy just in case some element is removed while iterating
                for callback in callbacks[:]:
                    callback(*args)
            else:
                print(f"No callback registered to event {event.name}")





