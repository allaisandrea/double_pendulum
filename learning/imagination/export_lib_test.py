import json

import numpy as np
import torch

from imagination.agent_lib import Agent
from imagination.env_lib import action_levels
from imagination.export_lib import export_policy
from world_model.model_lib import STEP_FEATURES


def test_the_export_holds_the_actor_and_cases_it_reproduces(tmp_path):
    torch.manual_seed(0)
    window, levels = 3, action_levels(16, 9)
    agent = Agent(window * STEP_FEATURES, 8, 2, len(levels))
    path = tmp_path / "policy.json"
    export_policy(agent, levels, window, path, {"test": True}, cases=4)
    e = json.loads(path.read_text())
    assert e["policy_window"] == window and e["levels"] == levels.tolist()
    # The layers, run as the harness runs them, give the actor's output.
    x = torch.randn(window * STEP_FEATURES)
    h = x.numpy()
    for i, layer in enumerate(e["layers"]):
        h = np.array(layer["weight"]) @ h + np.array(layer["bias"])
        if i + 1 < len(e["layers"]):
            h = np.tanh(h)
    assert np.allclose(h, agent.actor(x[None])[0].detach().numpy(), atol=1e-5)
    for case in e["cases"]:
        assert len(case["history"]) > window
        assert case["history"][0]["action"] is None
        assert case["action"] == levels[int(np.argmax(case["logits"]))]
