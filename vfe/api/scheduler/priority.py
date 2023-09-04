import atexit
from functools import partial
import logging
import os
import psutil
import queue
import threading
from torch import multiprocessing

from vfe import core
from .abstract import AbstractScheduler, Priority, PriorityTask

class PriorityScheduler(AbstractScheduler):
    @staticmethod
    def initialize_pool_with_pid(pid_queue):
        pid_queue.put(os.getpid())
        core.logging.configure_logger()

    def __init__(self, cpus: int, gpus: int, suspend_lowp=True):
        self.cpus = cpus
        self.gpus = gpus
        self.logger = logging.getLogger(__name__)
        self._should_suspend_lowp = suspend_lowp

        self._ctx = multiprocessing.get_context('spawn')
        pid_queue = self._ctx.Queue()
        self._pool = self._ctx.Pool(self.cpus, initializer=self.initialize_pool_with_pid, initargs=(pid_queue,))
        self._main_pool_processes = []
        for _ in range(self.cpus):
            self._main_pool_processes.append(psutil.Process(pid_queue.get()))

        self._lowp_gpu_pool_processes = []
        self._high_gpu_processes = []
        if self.gpus:
            self._highp_gpu_pool = self._ctx.Pool(1, initializer=self.initialize_pool_with_pid, initargs=(pid_queue,))
            self._high_gpu_processes.append(psutil.Process(pid_queue.get()))
            self._lowp_gpu_pool = self._ctx.Pool(self.gpus, initializer=self.initialize_pool_with_pid, initargs=(pid_queue,))
            for _ in range(self.gpus):
                self._lowp_gpu_pool_processes.append(psutil.Process(pid_queue.get()))

        self._resource_lock = threading.Lock()
        self._queued_tasks = queue.PriorityQueue()
        self._gpu_queued_tasks = queue.PriorityQueue()
        self._task_id = 0
        self._cpu_available_event = threading.Event()
        # Initially all cpus are available.
        self._cpu_available_event.set()
        self._used_resources = {
            'cpu': 0,
            'gpu': 0,
        }

        # Start thread after everything else is initialized.
        self._enqueue_thread = threading.Thread(group=None, target=self._enqueue_tasks, name='enqueue-tasks')
        self._enqueue_thread.daemon = True # Infinite loop never joins. Don't block shutdown on this.
        self._enqueue_thread.start()

        atexit.register(self.shutdown)

    def context(self):
        return self._ctx

    def shutdown(self):
        self.logger.info('Shutting down pool')
        self._pool.terminate()

        if self.gpus:
            self._lowp_gpu_pool.terminate()
            self._highp_gpu_pool.terminate()
        # self._process_args_queue.close()
        # Enqueue_thread won't stop until main process exits. If we want it to stop when shutdown
        # is called, we'll need to add an event to break out of the while True loop.

    def _enqueue_tasks(self):
        count = 0
        while True:
            count += 1
            if count and (count % 2000) == 0:
                self.logger.debug(f'Enqueue loop: {count}')

            # Assume task will just use one cpu.
            # self.logger.debug(f'Wait for available cpu')
            self._cpu_available_event.wait()

            # First check if the task that just ended was a GPU task.
            # If a GPU is now free and there are tasks on the GPU queue,
            # schedule them first because they were already bumped from the normal
            # queue.
            with self._resource_lock:
                gpu_available = self._used_resources['gpu'] < self.gpus

            gpu_task = None
            any_task = None
            if gpu_available and not self._gpu_queued_tasks.empty():
                gpu_task = self._gpu_queued_tasks.get()
                next_task = gpu_task

            # If gpu_task is None, we need to wait for some task.
            # If gpu_task is not None, we only need to get a task if one exists to compare priorities.
            if gpu_task is None or not self._queued_tasks.empty():
                # If we get past the wait, a cpu is available. The cpu should still be available when we
                # fetch a task because nothing else wil get scheduled without going through this loop.
                # Wait to pick a task until a cpu is available to avoid picking a task, waiting, and having
                # a higher-priority task come in while waiting for a cpu.
                # Time out after 0.5 seconds; it's possible in the mean time a gpu was freed up so we
                # can pull tasks from the gpu queue.
                try:
                    any_task = self._queued_tasks.get(block=True, timeout=0.5)
                except queue.Empty:
                    continue
                next_task = any_task

            # Check which task has priority; requeue the other.
            # If both are not set, next_task will be equal to the one that did have an item.
            if gpu_task and any_task:
                if gpu_task < any_task:
                    next_task = gpu_task
                    self._queued_tasks.put(any_task)
                else:
                    next_task = any_task
                    self._gpu_queued_tasks.put(gpu_task)

            fn, callback = next_task.item

            # Feature extraction needs a gpu; limit that separately.
            if next_task.name == 'extract_features':
                with self._resource_lock:
                    if self._used_resources['gpu'] >= self.gpus:
                        # Requeue; there isn't a gpu available.
                        # The task will be sorted back to the top because of its taskid.
                        # self.logger.debug(f'Requeueing task {next_task.name} ({next_task.task_id}) because no available gpu')
                        self._gpu_queued_tasks.put(next_task)
                        continue

            self.logger.debug(f'cpu available: {next_task.name} ({next_task.task_id}); Priority {next_task.priority}')
            with self._resource_lock:
                if self._used_resources['cpu'] >= self.cpus:
                    self.logger.warn(f'Warning: too many cpus allocated. This can happen if a high-priority task comes in. {self._used_resources["cpu"]} >= {self.cpus}')
                self._used_resources['cpu'] += 1
                used_gpu = False
                if next_task.name == 'extract_features':
                    self._used_resources['gpu'] += 1
                    used_gpu = True
                if self._used_resources['cpu'] >= self.cpus:
                    self._cpu_available_event.clear()

            if next_task.prepare:
                next_task.prepare()

            (
                self._lowp_gpu_pool if used_gpu else self._pool
            ).apply_async(
                fn,
                callback=self._handle_task_done(callback, used_gpu),
                error_callback=self._handle_task_error(callback, used_gpu),
            )

    def _handle_task_done(self, callback, used_gpu):
        def wrapped_callback(*args):
            with self._resource_lock:
                self._used_resources['cpu'] -= 1
                if used_gpu:
                    self._used_resources['gpu'] -= 1
                assert self._used_resources['cpu'] >= 0, f'{self._used_resources["cpu"]}'
                assert self._used_resources['gpu'] >= 0, f'{self._used_resources["gpu"]}'
                self._cpu_available_event.set()
            if callback is not None:
                callback(*args)
        return wrapped_callback

    def _handle_task_error(self, callback, used_gpu):
        def wrapped_callback(error):
            self.logger.exception(f"Error in task: {error}")
            with self._resource_lock:
                self._used_resources['cpu'] -= 1
                if used_gpu:
                    self._used_resources['gpu'] -= 1
                assert self._used_resources['cpu'] >= 0, f'{self._used_resources["cpu"]}'
                assert self._used_resources['gpu'] >= 0, f'{self._used_resources["gpu"]}'
                self._cpu_available_event.set()
            if callback is not None:
                callback()
        return wrapped_callback

    def _schedule_highp_gpu(self, name, fn, callback=None):
        normal_path = False
        with self._resource_lock:
            if self._used_resources['gpu'] < self.gpus:
                # Normal path if there isn't a gpu task already running, except we queue to the
                # high priority pool.
                self._used_resources['gpu'] += 1
                self._used_resources['cpu'] += 1 # TODO: wait for cpu.
                normal_path = True

        if normal_path:
            self._highp_gpu_pool.apply_async(fn, callback=self._handle_task_done(callback, True))
        else:
            # If a gpu task is already running, suspend it and start ours. Don't go through the
            # normal process of updating resources because the low_p process will do that once
            # it's resumed and finished. This assumes the running gpu task is a low-priority one.
            self.suspend_lowp()

            def wrapped_callback(*args, callback=None):
                self.resume_lowp()
                if callback is not None:
                    callback(*args)

            self._highp_gpu_pool.apply_async(fn, callback=partial(wrapped_callback, callback=callback))

    def schedule(self, name, fn, callback=None, priority: Priority = Priority.DEFAULT, prepare=None):
        if name == 'extract_features' and priority == 0:
            # Highest priority = 0 (assuming no negative values).
            self._schedule_highp_gpu(name, fn, callback)
        else:
            # Task id isn't threadsafe.
            task_id = self._task_id
            self._task_id += 1
            self.logger.debug(f'Queueing task {name} ({task_id})')
            self._queued_tasks.put(PriorityTask(priority, task_id, (fn, callback), name, prepare))

    def tasks_are_waiting(self):
        return not self._queued_tasks.empty() or not self._gpu_queued_tasks.empty()

    def suspend_lowp(self):
        # It doesn't seem to be an issue to suspend an already-stopped process.
        if self._should_suspend_lowp:
            for low_p in self._lowp_gpu_pool_processes:
                low_p.suspend()

    def resume_lowp(self):
        # It doesn't seem to be an issue to resume an already-running process.
        if self._should_suspend_lowp:
            for low_p in self._lowp_gpu_pool_processes:
                low_p.resume()

    def suspend_all(self):
        for processes in [self._main_pool_processes, self._lowp_gpu_pool_processes, self._high_gpu_processes]:
            for process in processes:
                process.suspend()

    def resume_all(self):
        for processes in [self._main_pool_processes, self._lowp_gpu_pool_processes, self._high_gpu_processes]:
            for process in processes:
                process.resume()
