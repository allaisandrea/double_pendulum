from world_model.profile import parse_log

LOG = """job run profile-od-g6 started at Sun Sep 27 01:44:54 UTC 2026 on g6.xlarge
NVIDIA L4, 23034 MiB
cuda: 2212899 training windows
 width  depth   batch    params       variant   ms/step  k samples/s  gather
   256      3    1024    175113         plain      2.07        495.4      5%
  1024      5   16384   4372489  compile+bf16     15.11       1084.6      1%
one evaluation of val_random_walk (4096 windows, 125 frames, width 256): 0.06 s
"""


def test_parse_log_reads_the_step_times_and_the_gpu():
    assert parse_log(LOG) == [
        ["NVIDIA L4", 256, 3, 1024, 175113, "plain", 2.07, 0.05],
        ["NVIDIA L4", 1024, 5, 16384, 4372489, "compile+bf16", 15.11, 0.01],
    ]


def test_parse_log_falls_back_to_the_device_it_ran_on():
    log = "\n".join(line for line in LOG.splitlines() if "MiB" not in line)
    assert parse_log(log)[0][0] == "cuda"
