from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, List, Optional, Protocol, TypeVar

from ..error import new_error


class InMessage(Protocol):
    def i_thread(self) -> int:  # pragma: no cover - protocol
        ...


class OutMessage(Protocol):
    @classmethod
    def shutdown(cls):  # pragma: no cover - protocol
        ...


class TaskQueueObserver(Protocol):
    def going_to_start_queue(self) -> None: ...

    def going_to_send(self, out_message, i_task: int, i_thread: int) -> None: ...

    def have_received(self, in_message, i_task: int, i_thread: int) -> None: ...

    def nothing_more_to_send(self) -> None: ...

    def completed_queue(self) -> None: ...


I = TypeVar("I", bound=InMessage)
O = TypeVar("O")


class WorkerLauncher(Protocol[I, O]):
    def launch(self, in_queue: queue.Queue[I], out_queue: queue.Queue[O], i_thread: int) -> None:
        ...


@dataclass
class Threads(Generic[I, O]):
    in_queue: queue.Queue[I]
    out_queues: List[queue.Queue[O]]
    join_handles: List[threading.Thread]

    @classmethod
    def new(cls, launcher: WorkerLauncher[I, O], n_threads: int) -> "Threads[I, O]":
        in_queue: queue.Queue[I] = queue.Queue()
        out_queues: List[queue.Queue[O]] = []
        join_handles: List[threading.Thread] = []
        for i_thread in range(n_threads):
            out_queue: queue.Queue[O] = queue.Queue()
            thread = threading.Thread(
                target=launcher.launch, args=(in_queue, out_queue, i_thread), daemon=True
            )
            thread.start()
            out_queues.append(out_queue)
            join_handles.append(thread)
        return cls(in_queue=in_queue, out_queues=out_queues, join_handles=join_handles)

    def n_threads(self) -> int:
        return len(self.join_handles)

    def broadcast(self, out_message: O) -> None:
        for out_queue in self.out_queues:
            out_queue.put(out_message)

    def responses_from_all(self) -> list[I]:
        responses: list[Optional[I]] = [None] * self.n_threads()
        while any(resp is None for resp in responses):
            response = self.in_queue.get()
            i_thread = response.i_thread()
            responses[i_thread] = response
        return [resp for resp in responses if resp is not None]

    def task_queue(self, out_messages: Iterable[O], observer: TaskQueueObserver) -> list[I]:
        observer.going_to_start_queue()
        maybe_more_out = True
        waiting_for_in = False
        task_by_thread: list[Optional[int]] = [None] * self.n_threads()
        in_messages: list[Optional[I]] = []
        out_iter = enumerate(out_messages)

        while maybe_more_out or waiting_for_in:
            if maybe_more_out:
                while True:
                    try:
                        i_thread_out = task_by_thread.index(None)
                    except ValueError:
                        break
                    try:
                        i_task, out_message = next(out_iter)
                    except StopIteration:
                        maybe_more_out = False
                        observer.nothing_more_to_send()
                        break
                    observer.going_to_send(out_message, i_task, i_thread_out)
                    self.out_queues[i_thread_out].put(out_message)
                    task_by_thread[i_thread_out] = i_task
                    waiting_for_in = True

            if waiting_for_in:
                in_message = self.in_queue.get()
                i_thread_in = in_message.i_thread()
                i_task = task_by_thread[i_thread_in]
                if i_task is None:
                    raise new_error("Received message with no queued task")
                observer.have_received(in_message, i_task, i_thread_in)
                task_by_thread[i_thread_in] = None
                while len(in_messages) < i_task:
                    in_messages.append(None)
                if len(in_messages) == i_task:
                    in_messages.append(in_message)
                else:
                    in_messages[i_task] = in_message
                waiting_for_in = any(value is not None for value in task_by_thread)

        observer.completed_queue()
        return [msg for msg in in_messages if msg is not None]

    def close(self, shutdown_message: O) -> None:
        for i, out_queue in enumerate(self.out_queues):
            out_queue.put(shutdown_message)
        for i, join_handle in enumerate(self.join_handles):
            join_handle.join(timeout=5)

