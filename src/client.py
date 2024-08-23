import torch
import zmq

from .core import share_cuda_tensor


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://0.0.0.0:6000")

    tensor = torch.randn((1, 3), dtype=torch.float, device="cuda:0")
    metadata = share_cuda_tensor(tensor)
    serialized_metadata = metadata

    while True:
        sock.send_pyobj(serialized_metadata)
        sock.recv_string()


if __name__ == "__main__":
    main()
