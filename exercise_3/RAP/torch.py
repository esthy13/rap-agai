"""
Minimal torch stub for environments that don't have PyTorch installed.

The original RAP codebase calls `torch.distributed.barrier()` only inside
error-handling branches (raise Exception paths). For our single-process
API-based implementation those paths are never reached, so a no-op stub is
sufficient to satisfy the import.
"""


class _Distributed:
    def barrier(self):
        pass  # no-op for single-process inference


distributed = _Distributed()
