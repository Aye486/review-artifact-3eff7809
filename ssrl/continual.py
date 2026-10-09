import copy
import torch


def prepare_stage(student, previous_state=None):
    if previous_state is None:
        return None
    student.load_state_dict(previous_state, strict=True)
    return copy.deepcopy(student).eval().requires_grad_(False)


def train_step(student, reference, inputs, optimizer, criterion, clip_max_norm=None):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = student(inputs)
    if reference is not None:
        reference.eval().requires_grad_(False)
        with torch.no_grad():
            outputs["previous_representation"] = reference(inputs)["representation"]
    rec, topology, drift = criterion.components(outputs)
    loss = criterion.weight * (rec + criterion.topology_weight * topology
                              + criterion.drift_weight * drift)
    loss.backward()
    if clip_max_norm is not None:
        if clip_max_norm <= 0:
            raise ValueError("clip_max_norm must be positive")
        torch.nn.utils.clip_grad_norm_(student.parameters(), clip_max_norm)
    optimizer.step()
    return {"loss": loss.detach(), "reconstruction": rec.detach(),
            "topology": topology.detach(), "drift": drift.detach()}
