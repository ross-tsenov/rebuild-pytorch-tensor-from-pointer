from dataclasses import asdict, dataclass
from typing import TypedDict, cast

import torch

from torch.multiprocessing.reductions import rebuild_cuda_tensor as _rebuild_tensor


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Converts a PyTorch dtype to its string representation."""

    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a PyTorch dtype back to its corresponding dtype object."""

    return getattr(torch, dtype)


class SerializedCudaRebuildMetadata(TypedDict):
    """TypedDict representing a serializable version of CUDA tensor rebuild metadata."""

    dtype: str
    tensor_size: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: str
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: str
    ref_counter_offset: int
    event_handle: str
    event_sync_required: bool


@dataclass
class CudaRebuildMetadata:
    """Data class representing the metadata required to rebuild a CUDA tensor."""

    dtype: torch.dtype
    tensor_size: torch.Size
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_serialized_dict(cls, metadata: SerializedCudaRebuildMetadata) -> "CudaRebuildMetadata":
        """Creates a `CudaRebuildMetadata` instance from a serialized dictionary."""

        return cls(
            dtype=str_to_torch_dtype(metadata["dtype"]),
            tensor_size=torch.Size(metadata["tensor_size"]),
            tensor_stride=tuple(metadata["tensor_stride"]),
            tensor_offset=metadata["tensor_offset"],
            storage_device=metadata["storage_device"],
            storage_handle=bytes.fromhex(metadata["storage_handle"]),
            storage_size_bytes=metadata["storage_size_bytes"],
            storage_offset_bytes=metadata["storage_offset_bytes"],
            requires_grad=metadata["requires_grad"],
            ref_counter_handle=bytes.fromhex(metadata["ref_counter_handle"]),
            ref_counter_offset=metadata["ref_counter_offset"],
            event_handle=bytes.fromhex(metadata["event_handle"]),
            event_sync_required=metadata["event_sync_required"],
        )

    def to_serialized_dict(self) -> SerializedCudaRebuildMetadata:
        """Converts this `CudaRebuildMetadata` instance into a serializable dictionary."""

        metadata = asdict(self)
        metadata["dtype"] = torch_dtype_to_str(self.dtype)
        metadata["tensor_size"] = tuple(self.tensor_size)
        metadata["storage_handle"] = self.storage_handle.hex()
        metadata["ref_counter_handle"] = self.ref_counter_handle.hex()
        metadata["event_handle"] = self.event_handle.hex()
        return cast(SerializedCudaRebuildMetadata, metadata)


def share_cuda_tensor(tensor: torch.Tensor) -> CudaRebuildMetadata:
    """Shares the CUDA memory of a tensor and generates `CudaRebuildMetadata` for rebuilding the tensor later.

    Args:
        tensor (torch.Tensor): The CUDA tensor to share.

    Returns:
        CudaRebuildMetadata: Metadata required to rebuild the shared CUDA tensor.
    """

    storage = tensor._typed_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return CudaRebuildMetadata(
        dtype=tensor.dtype,
        tensor_size=tensor.size(),
        tensor_stride=tensor.stride(),
        tensor_offset=tensor.storage_offset(),
        storage_device=device,
        storage_handle=handle,
        storage_size_bytes=storage_size_bytes,
        storage_offset_bytes=storage_offset_bytes,
        requires_grad=tensor.requires_grad,
        ref_counter_handle=ref_counter_handle,
        ref_counter_offset=ref_counter_offset,
        event_handle=event_handle,
        event_sync_required=event_sync_required,
    )


def rebuild_cuda_tensor(metadata: CudaRebuildMetadata) -> torch.Tensor:
    """Rebuilds a CUDA tensor from the provided `CudaRebuildMetadata`.

    Args:
        metadata (CudaRebuildMetadata): The metadata required to rebuild the tensor.

    Returns:
        torch.Tensor: The rebuilt CUDA tensor.
    """

    return _rebuild_tensor(
        tensor_cls=torch.Tensor,
        tensor_size=metadata.tensor_size,
        tensor_stride=metadata.tensor_stride,
        tensor_offset=metadata.tensor_offset,
        storage_cls=torch.TypedStorage,
        dtype=metadata.dtype,
        storage_device=metadata.storage_device,
        storage_handle=metadata.storage_handle,
        storage_size_bytes=metadata.storage_size_bytes,
        storage_offset_bytes=metadata.storage_offset_bytes,
        requires_grad=metadata.requires_grad,
        ref_counter_handle=metadata.ref_counter_handle,
        ref_counter_offset=metadata.ref_counter_offset,
        event_handle=metadata.event_handle,
        event_sync_required=metadata.event_sync_required,
    )
