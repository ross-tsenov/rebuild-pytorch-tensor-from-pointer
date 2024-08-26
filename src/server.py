"""Server that receives tensor rebuilding metadata, reconstructs tensors,
generates new rebuilding metadata, and sends the results back.
"""

from typing import cast

import zmq

from .core import CudaRebuildMetadata, SerializedCudaRebuildMetadata, rebuild_cuda_tensor, share_cuda_tensor


def run_basic_server(sock: zmq.SyncSocket) -> None:
    """Runs a basic server that receives a pickled tensor and sends it back unchanged."""

    while True:
        tensor = sock.recv_pyobj()
        sock.send_pyobj(tensor)


def run_pyobj_server(sock: zmq.SyncSocket) -> None:
    """Runs a server that receives `CudaRebuildMetadata`, rebuilds the tensor
    using this metadata, re-shares the tensor by generating new `CudaRebuildMetadata`,
    and sends it back."""

    while True:
        cuda_rebuild_metadata = cast(CudaRebuildMetadata, sock.recv_pyobj())
        # cannot re-share same tensor, have to clone it. This only done for testing.
        rebuilt_tensor = rebuild_cuda_tensor(cuda_rebuild_metadata).clone()
        sock.send_pyobj(share_cuda_tensor(rebuilt_tensor))


def run_json_server(sock: zmq.SyncSocket) -> None:
    """Runs a server that receives `SerializedCudaRebuildMetadata` as a dictionary,
    rebuilds the tensor using this metadata, re-shares the tensor by generating new
    `SerializedCudaRebuildMetadata`, and sends it back as a dictionary."""

    while True:
        cuda_rebuild_metadata = cast(SerializedCudaRebuildMetadata, sock.recv_json())
        # cannot re-share same tensor, have to clone it. This only done for testing.
        rebuilt_tensor = rebuild_cuda_tensor(CudaRebuildMetadata.from_serialized_dict(cuda_rebuild_metadata)).clone()
        sock.send_json(share_cuda_tensor(rebuilt_tensor).to_serialized_dict())


def main() -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:6000")

    # Choose one and ensure the client file is using the same option.
    run_basic_server(sock)
    # run_pyobj_server(sock)
    # run_json_server(sock)


if __name__ == "__main__":
    main()
