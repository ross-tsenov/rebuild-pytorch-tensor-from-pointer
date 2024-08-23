import base64

from dataclasses import asdict, dataclass
from typing import TypedDict, cast

import torch

from torch.multiprocessing.reductions import rebuild_cuda_tensor


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)


class SerializedCudaRebuildMetadata(TypedDict):
    dtype: str
    tensor_size: tuple[int, ...]
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


@dataclass
class CudaRebuildMetadata:
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
        return cls(
            dtype=str_to_torch_dtype(metadata["dtype"]),
            tensor_size=torch.Size(metadata["tensor_size"]),
            tensor_stride=tuple(metadata["tensor_stride"]),
            tensor_offset=metadata["tensor_offset"],
            storage_device=metadata["storage_device"],
            storage_handle=base64.b64decode(metadata["storage_handle"]),
            storage_size_bytes=metadata["storage_size_bytes"],
            storage_offset_bytes=metadata["storage_offset_bytes"],
            requires_grad=metadata["requires_grad"],
            ref_counter_handle=base64.b64decode(metadata["ref_counter_handle"]),
            ref_counter_offset=metadata["ref_counter_offset"],
            event_handle=base64.b64decode(metadata["event_handle"]),
            event_sync_required=metadata["event_sync_required"],
        )

    def to_serialized_dict(self) -> SerializedCudaRebuildMetadata:
        metadata = asdict(self)
        metadata["dtype"] = torch_dtype_to_str(self.dtype)
        metadata["tensor_size"] = tuple(self.tensor_size)
        metadata["storage_handle"] = base64.b64encode(self.storage_handle)
        metadata["ref_counter_handle"] = base64.b64encode(self.ref_counter_handle)
        metadata["event_handle"] = base64.b64encode(self.event_handle)
        return cast(SerializedCudaRebuildMetadata, metadata)


def share_cuda_tensor(tensor: torch.Tensor) -> CudaRebuildMetadata:
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


def restore_cuda_tensor(metadata: CudaRebuildMetadata) -> torch.Tensor:
    return rebuild_cuda_tensor(
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
