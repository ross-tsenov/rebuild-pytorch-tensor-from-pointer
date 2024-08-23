import time

import zmq

from .core import restore_cuda_tensor


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:6000")

    for i in range(10):
        before = time.time()

        cuda_rebuild_metadata = sock.recv_pyobj()
        rebuilt_tensor = restore_cuda_tensor(cuda_rebuild_metadata)

        after = time.time()
        print(after - before)

        sock.send_string("")


if __name__ == "__main__":
    main()
