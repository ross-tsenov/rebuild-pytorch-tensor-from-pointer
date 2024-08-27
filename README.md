# Rebuild Tensor From Reference Sample
This repo contains sample code and usage instructions on how you can share PyTorch tensor pointers/references between processes running on the same system and using the same GPU, instead of copying the entire tensor between processes through the CPU. This approach can save significant time. Let’s just say, at one point, I went a little crazy with microservices architecture and AI processing. In the end, we scrapped the idea at work because it wasn't worth the effort, and unfortunately, it doesn’t play well with Jetsons. But hey, I wanted to share it here in case someone else falls down the same rabbit hole.

The code heavily relies on the use of "private" PyTorch functions and methods, if you will. So, I highly **do not recommend** using this in any large-scale production environment—use it just for fun or experimentation.

### Overview

There’s a client and a server. The server essentially takes in a tensor pointer and some additional metadata, rebuilds it, and sends it back. The client generates a random tensor, sends it to the server, rebuilds the tensor from the received pointer, and compares it to make sure the tensor hasn’t changed. It also checks the time it took for the round trip. Additionally, there’s `src/core.py`, which contains core functions used by both the client and server. Specifically, it includes:

- `CudaRebuildMetadata`: A dataclass that describes everything you need to rebuild a tensor from its pointer.
- `SerializedCudaRebuildMetadata`: A JSON-serializable dictionary as an alternative to the dataclass.
- `share_cuda_tensor`: A function that takes in a tensor, shares its memory, and generates `CudaRebuildMetadata`.
- `rebuild_cuda_tensor`: A function that takes in `CudaRebuildMetadata` and rebuilds the tensor. That’s about it.

### Experiment

I experimented with sending a fully pickled tensor and just metadata to the server, receiving it back, and tracking the timing for the round trip of the tensor to and from the server over a TCP connection on localhost. The experiment ran over 100,000 iterations, and the average time for each alternative was compared. Note that the tensor size was `(1920, 1080, 3)`, and the dtype was `torch.float`."

- **Sending a full pickled tensor** took about 110ms on average.
- **Sending the Dataclass** took 0.89ms on average, with 0.02ms taken by the sharing function, 0.13ms taken by the rebuilding function, and 0.24ms taken by the cloning function (this was necessary because we were re-sharing the same tensor).
- **Sending the Serialized Dataclass** took 0.97ms, slightly slower, probably due to the serialization and deserialization overhead.

So, you could achieve more than a 100x improvement this way if you need to send tensors between processes.

### Conclusion

You can see that with just some custom code and data types, you can easily improve the transfer speed of a tensor if it’s located on the same GPU, that is. If sub-1ms transfer times are still too slow for you, well, maybe you shouldn't be using Python in the first place. If you'd like to use my code, feel free; everything you need is located in `src/core.py`.
