"""A branch adapter must reload exactly and reject a different base revision."""

import tempfile
from pathlib import Path

import torch

from ba_dit.checkpoint import load_branch, save_branch
from ba_dit.nn.reference_read_delta import ReferenceReadDelta


def test_branch_checkpoint_round_trip():
    model = torch.nn.Sequential(ReferenceReadDelta(8, 2, rank=2))
    identity = {"backend": "tiny", "model_revision": "rev1", "source_commit": "pin1"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "branch.safetensors"
        save_branch(model, path, identity)
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        try:
            load_branch(model, path, {**identity, "model_revision": "rev2"})
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong base revision was accepted")
        load_branch(model, path, identity)
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, original[name])
