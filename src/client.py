"""Client that generates a random tensor and either shares the full tensor or its reconstruction metadata with a server.
After receiving new information back from the server, it rebuilds the tensor and compares it with the original
to ensure they are identical. The script also measures the time taken for the round trip and tensor rebuilding."""

import time

from typing import cast

import torch
import zmq

from .core import CudaRebuildMetadata, SerializedCudaRebuildMetadata, rebuild_cuda_tensor, share_cuda_tensor


def basic_client(sock: zmq.SyncSocket, tensor: torch.Tensor) -> torch.Tensor:
    """Sends the entire tensor object to the server and receives the rebuilt tensor from the server."""

    sock.send_pyobj(tensor)
    received_tensor = sock.recv_pyobj()

    return received_tensor


def dataclass_client(sock: zmq.SyncSocket, tensor: torch.Tensor) -> torch.Tensor:
    """Shares CUDA tensor metadata with the server and receives the rebuilt tensor."""

    sock.send_pyobj(share_cuda_tensor(tensor))
    cuda_rebuild_metadata = sock.recv_pyobj()
    received_tensor = rebuild_cuda_tensor(cuda_rebuild_metadata)

    return received_tensor


def json_client(sock: zmq.SyncSocket, tensor: torch.Tensor) -> torch.Tensor:
    """Shares CUDA tensor metadata with the server in serialized JSON format and receives the rebuilt tensor."""

    sock.send_json(share_cuda_tensor(tensor).to_serialized_dict())
    cuda_rebuild_metadata = cast(SerializedCudaRebuildMetadata, sock.recv_json())
    received_tensor = rebuild_cuda_tensor(CudaRebuildMetadata.from_serialized_dict(cuda_rebuild_metadata))

    return received_tensor


def main() -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://0.0.0.0:6000")

    latencies: list[float] = []
    for _ in range(1000):
        tensor_to_send = torch.randn((1920, 1080, 3), dtype=torch.float, device="cuda:0")

        before = time.perf_counter()
        # Choose one client implementation and ensure the server is using the same approach.
        # received_tensor = basic_client(sock, tensor_to_send)
        # received_tensor = dataclass_client(sock, tensor_to_send)
        received_tensor = json_client(sock, tensor_to_send)
        after = time.perf_counter()
        latency = after - before
        latencies.append(latency)

        assert tensor_to_send.size() == received_tensor.size()
        assert bool(torch.all(tensor_to_send == received_tensor))

    latencies = latencies[1:]  # Skip first one, due to slow execution.
    average_latency = sum(latencies) / len(latencies)
    print(f"Average latency of every request is {1000*average_latency:.3f}ms.")


if __name__ == "__main__":
    main()
