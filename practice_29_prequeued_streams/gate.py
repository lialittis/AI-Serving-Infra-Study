"""One CANN Notify per compute stream; a separate native stream opens the gate.

Borrow torch-npu's initialized context. Never initialize/reset the device here.
All native handles are resolved BEFORE closing the gate. The watchdog submits
only control notifications (never model work), bypassing torch-npu's task queue.
"""
import ctypes as C
import threading
import time


class Gate:
    def __init__(self, streams, timeout=2.0):
        self.lib = C.CDLL('libascendcl.so')
        signatures = {
            'aclrtGetCurrentContext': [C.POINTER(C.c_void_p)],
            'aclrtSetCurrentContext': [C.c_void_p],
            'aclrtCreateStream': [C.POINTER(C.c_void_p)],
            'aclrtDestroyStream': [C.c_void_p],
            'aclrtCreateNotify': [C.POINTER(C.c_void_p), C.c_uint64],
            'aclrtDestroyNotify': [C.c_void_p],
            'aclrtWaitAndResetNotify': [C.c_void_p, C.c_void_p, C.c_uint32],
            'aclrtRecordNotify': [C.c_void_p, C.c_void_p],
            'aclrtStreamQuery': [C.c_void_p, C.POINTER(C.c_int)],
            'aclrtSynchronizeStreamWithTimeout': [C.c_void_p, C.c_int32],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.context, self.release_stream = C.c_void_p(), C.c_void_p()
        self.call('aclrtGetCurrentContext', C.byref(self.context))
        self.call('aclrtCreateStream', C.byref(self.release_stream))
        self.handles = [C.c_void_p(s.npu_stream) for s in streams]
        self.notifies = []
        self.armed, self.opened, self.rescue = [], False, False
        self.lock, self.log, self.timeout = threading.Lock(), [], timeout
        self.timer = None
        try:
            for _ in streams:
                notify = C.c_void_p()
                self.call('aclrtCreateNotify', C.byref(notify), 0)
                self.notifies.append(notify)
        except BaseException:
            self.destroy()
            raise

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc:
            raise RuntimeError(f'{name}: {rc}')

    def stamp(self, action, **extra):
        self.log.append(dict(action=action, monotonic_ns=time.perf_counter_ns(), **extra))

    def close(self):
        # Each object is single use. A consumed Notify automatically resets.
        self.timer = threading.Timer(self.timeout, self.emergency_open)
        self.timer.daemon = True
        self.timer.start()
        with self.lock:
            if self.opened:
                raise RuntimeError('gate watchdog expired before arming')
            for i, (notify, stream) in enumerate(zip(self.notifies, self.handles)):
                self.armed.append(i)  # Also release a possibly submitted failed call.
                self.call('aclrtWaitAndResetNotify', notify, stream, 0)
                self.stamp('wait_enqueued', index=i)

    def open(self, rescue=False):
        with self.lock:
            if self.opened:
                return
            self.call('aclrtSetCurrentContext', self.context)
            self.rescue = rescue
            self.stamp('release_begin', rescue=rescue)
            for i in self.armed:
                self.call('aclrtRecordNotify', self.notifies[i], self.release_stream)
            self.stamp('release_return', rescue=rescue)
            self.opened = True
        if self.timer:
            self.timer.cancel()

    def emergency_open(self):
        try:
            self.open(rescue=True)
        except BaseException as exc:
            self.stamp('rescue_failed', error=repr(exc))

    def query(self):
        values = []
        for stream in self.handles:
            status = C.c_int()
            self.call('aclrtStreamQuery', stream, C.byref(status))
            values.append(status.value)  # Installed header: COMPLETE=0, NOT_READY=1.
        self.stamp('query', status=values)
        return values

    def destroy(self):
        if self.armed:
            self.open()
        if self.timer:
            self.timer.cancel()
            self.timer.join(timeout=3)
        # Caller must drain torch-npu queues BEFORE destroying native handles.
        for stream in [*self.handles, self.release_stream]:
            self.call('aclrtSynchronizeStreamWithTimeout', stream, 3000)
        for notify in self.notifies:
            self.call('aclrtDestroyNotify', notify)
        self.call('aclrtDestroyStream', self.release_stream)
