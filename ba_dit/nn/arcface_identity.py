"""Differentiable executor for the frozen InsightFace ArcFace ONNX graph."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenOnnxArcFace(nn.Module):
    """Execute buffalo_l ``w600k_r50.onnx`` while retaining input gradients."""

    SUPPORTED_OPS = {
        "Add",
        "BatchNormalization",
        "Conv",
        "Flatten",
        "Gemm",
        "PRelu",
    }

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        import onnx
        from onnx import numpy_helper

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ArcFace ONNX model does not exist: {path}")
        model_sha256 = sha256_file(path)
        if expected_sha256 and model_sha256 != str(expected_sha256).lower():
            raise RuntimeError(
                "ArcFace ONNX SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={model_sha256}, path={path}"
            )

        graph = onnx.load(str(path), load_external_data=True).graph
        if len(graph.input) != 1 or len(graph.output) != 1:
            raise RuntimeError("ArcFace executor requires one graph input and output")
        unsupported = sorted(
            {node.op_type for node in graph.node} - self.SUPPORTED_OPS
        )
        if unsupported:
            raise RuntimeError(f"Unsupported ArcFace ONNX operators: {unsupported}")

        self.model_path = str(path)
        self.model_sha256 = model_sha256
        self.input_name = graph.input[0].name
        self.output_name = graph.output[0].name
        self._initializer_buffers: dict[str, str] = {}
        for index, initializer in enumerate(graph.initializer):
            buffer_name = f"onnx_initializer_{index:03d}"
            value = torch.from_numpy(numpy_helper.to_array(initializer).copy())
            if not value.is_floating_point():
                raise RuntimeError(
                    "Unexpected non-floating ArcFace initializer "
                    f"{initializer.name!r}"
                )
            self.register_buffer(
                buffer_name,
                value.float(),
                persistent=False,
            )
            self._initializer_buffers[initializer.name] = buffer_name

        self._nodes = [
            {
                "op_type": node.op_type,
                "name": node.name,
                "inputs": tuple(node.input),
                "outputs": tuple(node.output),
                "attributes": {
                    attribute.name: onnx.helper.get_attribute_value(attribute)
                    for attribute in node.attribute
                },
            }
            for node in graph.node
        ]
        self.requires_grad_(False)
        self.eval()

    def _initializer(self, name: str) -> torch.Tensor:
        return getattr(self, self._initializer_buffers[name])

    @staticmethod
    def _attribute(node: dict[str, Any], name: str, default: Any) -> Any:
        return node["attributes"].get(name, default)

    def _resolve(
        self,
        values: dict[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        if name in values:
            return values[name]
        if name in self._initializer_buffers:
            return self._initializer(name)
        raise RuntimeError(f"ArcFace ONNX input {name!r} was not resolved")

    def _execute_node(
        self,
        node: dict[str, Any],
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        op_type = node["op_type"]
        if op_type == "Add":
            return inputs[0] + inputs[1]
        if op_type == "Conv":
            pads = tuple(
                int(value)
                for value in self._attribute(node, "pads", [0, 0, 0, 0])
            )
            if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
                raise RuntimeError(
                    f"Unsupported asymmetric Conv padding in {node['name']}"
                )
            strides = tuple(
                int(value)
                for value in self._attribute(node, "strides", [1, 1])
            )
            dilations = tuple(
                int(value)
                for value in self._attribute(node, "dilations", [1, 1])
            )
            group = int(self._attribute(node, "group", 1))
            bias = inputs[2] if len(inputs) == 3 else None
            return F.conv2d(
                inputs[0],
                inputs[1],
                bias,
                stride=strides,
                padding=(pads[0], pads[1]),
                dilation=dilations,
                groups=group,
            )
        if op_type == "PRelu":
            slope = inputs[1]
            if slope.numel() == 1:
                slope = slope.reshape(*([1] * inputs[0].ndim))
            elif slope.numel() == inputs[0].shape[1]:
                slope = slope.reshape(
                    1,
                    inputs[0].shape[1],
                    *([1] * (inputs[0].ndim - 2)),
                )
            else:
                raise RuntimeError(
                    f"Unsupported PRelu slope shape {tuple(slope.shape)} "
                    f"in {node['name']}"
                )
            return torch.where(inputs[0] >= 0, inputs[0], inputs[0] * slope)
        if op_type == "BatchNormalization":
            epsilon = float(self._attribute(node, "epsilon", 1.0e-5))
            return F.batch_norm(
                inputs[0],
                inputs[3],
                inputs[4],
                weight=inputs[1],
                bias=inputs[2],
                training=False,
                momentum=0.0,
                eps=epsilon,
            )
        if op_type == "Flatten":
            axis = int(self._attribute(node, "axis", 1))
            if axis < 0:
                axis += inputs[0].ndim
            leading = 1
            trailing = 1
            for size in inputs[0].shape[:axis]:
                leading *= int(size)
            for size in inputs[0].shape[axis:]:
                trailing *= int(size)
            return inputs[0].reshape(leading, trailing)
        if op_type == "Gemm":
            left = (
                inputs[0].transpose(-1, -2)
                if int(self._attribute(node, "transA", 0))
                else inputs[0]
            )
            right = (
                inputs[1].transpose(-1, -2)
                if int(self._attribute(node, "transB", 0))
                else inputs[1]
            )
            result = float(self._attribute(node, "alpha", 1.0)) * torch.matmul(
                left,
                right,
            )
            if len(inputs) == 3:
                result = result + float(
                    self._attribute(node, "beta", 1.0)
                ) * inputs[2]
            return result
        raise RuntimeError(f"Unhandled ArcFace ONNX operator {op_type!r}")

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        if normalized_rgb.ndim != 4 or normalized_rgb.shape[1:] != (3, 112, 112):
            raise ValueError(
                "ArcFace input must be normalized RGB [N,3,112,112], got "
                f"{tuple(normalized_rgb.shape)}"
            )
        values = {self.input_name: normalized_rgb.float()}
        for node in self._nodes:
            inputs = [
                self._resolve(values, name)
                for name in node["inputs"]
                if name
            ]
            output = self._execute_node(node, inputs)
            if len(node["outputs"]) != 1:
                raise RuntimeError(
                    f"ArcFace node has multiple outputs: {node['name']}"
                )
            values[node["outputs"][0]] = output
        return values[self.output_name]
