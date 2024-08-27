"""Server that receives tensor rebuilding metadata, reconstructs tensors,
generates new rebuilding metadata, and sends the results back.
"""

from typing import cast

import zmq

from src import settings
from src.core import (
    CudaRebuildMetadata,
    SerializedCudaRebuildMetadata,
    rebuild_cuda_tensor,
    share_cuda_tensor,
)


def dataclass_metadata_server(sock: zmq.SyncSocket) -> None:
    """Runs a server that receives `CudaRebuildMetadata`, rebuilds the tensor
    using this metadata, re-shares the tensor by generating new `CudaRebuildMetadata`,
    and sends it back."""

    while True:
        cuda_rebuild_metadata = cast(CudaRebuildMetadata, sock.recv_pyobj())
        # cannot re-share same tensor, have to clone it. This only done for testing.
        rebuilt_tensor = rebuild_cuda_tensor(cuda_rebuild_metadata).clone()
        sock.send_pyobj(share_cuda_tensor(rebuilt_tensor))


def json_metadata_server(sock: zmq.SyncSocket) -> None:
    """Runs a server that receives `SerializedCudaRebuildMetadata` as a dictionary,
    rebuilds the tensor using this metadata, re-shares the tensor by generating new
    `SerializedCudaRebuildMetadata`, and sends it back as a dictionary."""

    while True:
        cuda_rebuild_metadata = cast(SerializedCudaRebuildMetadata, sock.recv_json())
        # cannot re-share same tensor, have to clone it. This only done for testing.
        rebuilt_tensor = rebuild_cuda_tensor(CudaRebuildMetadata.from_serialized_dict(cuda_rebuild_metadata)).clone()
        sock.send_json(share_cuda_tensor(rebuilt_tensor).to_serialized_dict())


def full_tensor_server(sock: zmq.SyncSocket) -> None:
    """Runs a basic server that receives a pickled tensor and sends it back unchanged."""

    while True:
        tensor = sock.recv_pyobj()
        sock.send_pyobj(tensor)


RUNNING_TYPE_TO_FUNC = {
    settings.RunningType.FULL_TENSOR: full_tensor_server,
    settings.RunningType.DATACLASS_METADATA: dataclass_metadata_server,
    settings.RunningType.JSON_METADATA: json_metadata_server,
}


def main() -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"{settings.SERVER_PROTOCOL}://*:{settings.SERVER_PORT}")

    RUNNING_TYPE_TO_FUNC[settings.RUNNING_TYPE](sock)


if __name__ == "__main__":
    main()
