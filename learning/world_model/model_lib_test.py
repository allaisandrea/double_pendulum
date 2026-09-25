import torch

from world_model.model_lib import last_seen


def test_last_seen_takes_each_tags_latest_observation():
    obs = torch.arange(1, 4, dtype=torch.float32)[None, :, None, None].expand(1, 3, 3, 2)
    present = torch.tensor([[[True, True, False], [True, False, False], [False, False, False]]])
    ref, seen = last_seen(obs, present)
    assert ref[0, :, 0].tolist() == [2, 1, 0]
    assert seen[0].tolist() == [True, True, False]
