# Recordings

Every recording uploaded to `s3://allais-andrea-store/double_pendulum/data/`, newest first,
written by `uv run python -m docs.recordings` from the recordings themselves. On
2026-09-30 a tag on the rig moved: recordings from `1790789841` on are of the rig as
it is now, the earlier ones of the rig as it was, and the two are not mixed.

- **kind**: a random walk (the stand-in policy), a collection with a trained policy
  (training data), or a greedy test (named by its ranking manifest in `recordings/`).
- **actions**: how the policy acted (greedy, sampled, with exploration bursts and
  their offset and clip), then its number of actions and their range.
- **drive/rest s**: the duty cycle, seconds driving then resting.
- **tags seen %**: the share of frames each of tags 0, 1 and 2 was seen in.
- **used in**: by role (`train`, a validation set, or `starts`, a policy's
  episode starts), the configs that name the recording (`wm:` world model,
  `policy:` policy) and the cloud runs that name it with `--set`, as the loops'
  runs do (those in `runs/` here).

## The rig as it is now

23 recordings, 6.0 hours.

### 2026-10-01

3 recordings, 0.7 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790872817` | 10-01 09:40 | 5 | greedy test (ranking-1790872801-seed1) | ppo-rb2-it1 | greedy; 29 to ±126 | 20/8 | 100/97/100 |  |
| `1790870406` | 10-01 09:00 | 30 | collection | ppo-rb2-it2 | sampled; 29 to ±126 | 20/8 | 100/97/100 |  |
| `1790870082` | 10-01 08:54 | 5 | greedy test (ranking-1790870067-seed0) | ppo-rb2-it2 | greedy; 29 to ±126 | 20/8 | 99/96/100 |  |

### 2026-09-30

20 recordings, 5.3 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790828159` | 09-30 21:16 | 16 | collection | ppo-rb2-it1 | sampled; 29 to ±126 | 20/8 | 100/97/100 |  |
| `1790826915` | 09-30 20:55 | 20 | collection | ppo-rb2-it1 | sampled; 29 to ±126 | 20/8 | 100/97/100 |  |
| `1790825675` | 09-30 20:34 | 20 | collection | ppo-rb2-it1 | sampled; 29 to ±126 | 20/8 | 100/97/100 |  |
| `1790825348` | 09-30 20:29 | 5 | greedy test (ranking-1790825332-seed0) | ppo-rb2-it1 | greedy; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790824091` | 09-30 20:08 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/99 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790822847` | 09-30 19:47 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/96/99 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790821607` | 09-30 19:26 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790820333` | 09-30 19:05 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/96/99 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790819094` | 09-30 18:44 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/96/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790817862` | 09-30 18:24 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/99 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790816632` | 09-30 18:03 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790815402` | 09-30 17:43 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790814173` | 09-30 17:22 | 20 | collection | ppo-rb-it3 | sampled; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it2; train: rb2-it2 |
| `1790801535` | 09-30 13:52 | 5 | greedy test (ranking-1790801519-seed0) | ppo-rb-it3 | greedy; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it1, ppo-rb2-it2; train: rb2-it1, rb2-it2 |
| `1790798133` | 09-30 12:55 | 20 | collection | ppo-rb-it2 | sampled; 29 to ±126 | 20/8 | 100/96/100 | starts: ppo-rb-it3, ppo-rb2-it1, ppo-rb2-it2; train: rb-it3, rb2-it1, rb2-it2 |
| `1790797815` | 09-30 12:50 | 5 | greedy test (ranking-1790797799-seed0) | ppo-rb-it2 | greedy; 29 to ±126 | 20/8 | 100/97/100 | starts: ppo-rb2-it1, ppo-rb2-it2; train: rb2-it1, rb2-it2 |
| `1790794013` | 09-30 11:46 | 20 | collection | ppo-rb-it1 | sampled; 29 to ±126 | 20/8 | 96/97/99 | starts: ppo-rb-it2, ppo-rb-it3, ppo-rb2-it1, ppo-rb2-it2; train: rb-it2, rb-it3, rb2-it1, rb2-it2 |
| `1790793694` | 09-30 11:41 | 5 | greedy test (ranking-1790793679-seed0) | ppo-rb-it1 | greedy; 29 to ±126 | 20/8 | 95/97/99 | starts: ppo-rb2-it1, ppo-rb2-it2; train: rb2-it1, rb2-it2 |
| `1790791100` | 09-30 10:58 | 5 | greedy test (ranking-1790791084-seed0) | ppo-s256-anneal0-5k | greedy; 15 to ±126 | 20/8 | 100/97/98 | starts: ppo-rb2-it1, ppo-rb2-it2; train: rb2-it1, rb2-it2 |
| `1790789841` | 09-30 10:37 | 20 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 20/8 | 100/97/98 | starts: ppo-rb-it1, ppo-rb-it2, ppo-rb-it3, ppo-rb2-it1, ppo-rb2-it2; train: rb-it1, rb-it2, rb-it3, rb2-it1, rb2-it2 |

## Before the tag moved (2026-09-30)

67 recordings, 16.2 hours.

### 2026-09-29

6 recordings, 3.9 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790731399` | 09-29 18:23 | 25 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/96/100 | train: wm:stochastic-3 |
| `1790727759` | 09-29 17:22 | 60 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/97/99 | train: wm:stochastic-3 |
| `1790725492` | 09-29 16:44 | 35 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/97/99 | train: wm:stochastic-3 |
| `1790721853` | 09-29 15:44 | 60 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/97/100 | train: wm:stochastic-3 |
| `1790716300` | 09-29 14:11 | 50 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3 |
| `1790715979` | 09-29 14:06 | 5 | collection | ppo-s256-anneal0-5k | sampled; 15 to ±126 | 30/8 | 100/97/100 | val_long_drives: wm:stochastic-2, wm:stochastic-3 |

### 2026-09-28

10 recordings, 5.4 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790645280` | 09-28 18:28 | 60 | collection | ppo-stoch-it4 | sampled; 15 to ±126 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it5; starts: policy:wm-s256-anneal, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean |
| `1790644962` | 09-28 18:22 | 5 | collection | ppo-stoch-it4 | sampled; 15 to ±126 | 10/5 | 99/97/100 | val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic; val_ppo_stoch_it4: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790636499` | 09-28 16:01 | 60 | collection | ppo-stoch-it3 | sampled, bursts ±48 to ±127; 13 to ±96 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it4, stoch-it5; starts: policy:wm-s256-anneal, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it4 |
| `1790636181` | 09-28 15:56 | 5 | collection | ppo-stoch-it3 | sampled, bursts ±48 to ±127; 13 to ±96 | 10/5 | 99/97/100 | val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790627097` | 09-28 13:24 | 60 | collection | ppo-stoch-it2 | sampled; 13 to ±96 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it3, stoch-it4, stoch-it5; starts: policy:wm-s256-anneal, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it3, ppo-stoch-it4 |
| `1790626779` | 09-28 13:19 | 5 | collection | ppo-stoch-it2 | sampled; 13 to ±96 | 10/5 | 99/97/100 | val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790619529` | 09-28 11:18 | 60 | collection | ppo-stoch-it1 | sampled; 13 to ±96 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:wm-s256-anneal, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790619211` | 09-28 11:13 | 5 | collection | ppo-stoch-it1 | sampled; 13 to ±96 | 10/5 | 99/97/100 | val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790611952` | 09-28 09:12 | 60 | collection | ppo-stoch | sampled, bursts ±48 to ±96; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:wm-s256-anneal, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790611634` | 09-28 09:07 | 5 | collection | ppo-stoch | sampled, bursts ±48 to ±96; 9 to ±64 | 10/5 | 99/97/100 | val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |

### 2026-09-27

42 recordings, 1.4 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790548419` | 09-27 15:33 | 2 | greedy test (ranking-1790548404-seed1) | ppo-stoch | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790548281` | 09-27 15:31 | 2 | greedy test (ranking-1790548265-seed0) | ppo-stoch | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545913` | 09-27 14:51 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k1@2000 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545775` | 09-27 14:49 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k1-seed1 | greedy; 9 to ±64 | 10/5 | 100/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545637` | 09-27 14:47 | 2 | greedy test (ranking-1790545071-seed1) | ppo-wm3 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545500` | 09-27 14:45 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k1 | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545362` | 09-27 14:42 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k6@1000 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545224` | 09-27 14:40 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k6 | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790545086` | 09-27 14:38 | 2 | greedy test (ranking-1790545071-seed1) | ppo-k1-seed1@2000 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790544948` | 09-27 14:35 | 2 | greedy test (ranking-1790544933-seed0) | ppo-k1-seed1@2000 | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790543834` | 09-27 14:17 | 2 | greedy test (ranking-1790543406-seed0) | ppo-k6 | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790543696` | 09-27 14:14 | 2 | greedy test (ranking-1790543406-seed0) | ppo-k1@2000 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790543559` | 09-27 14:12 | 2 | greedy test (ranking-1790543406-seed0) | ppo-wm3 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790543421` | 09-27 14:10 | 2 | greedy test (ranking-1790543406-seed0) | ppo-k6@1000 | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790541999` | 09-27 13:46 | 2 | greedy test (ranking-1790541846-seed0) | ppo-k1 | greedy; 9 to ±64 | 10/5 | 100/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790541861` | 09-27 13:44 | 2 | greedy test (ranking-1790541846-seed0) | ppo-k1-seed1 | greedy; 9 to ±64 | 10/5 | 100/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790527070` | 09-27 09:37 | 2 | greedy test (ranking-1790525401-seed1) | ppo-wm2 | greedy; 9 to ±64 | 10/5 | 99/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526932` | 09-27 09:35 | 2 | greedy test (ranking-1790525401-seed1) | ppo-k8@250 | greedy; 9 to ±64 | 10/5 | 99/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526794` | 09-27 09:33 | 2 | greedy test (ranking-1790525401-seed1) | ppo-k8@100 | greedy; 9 to ±64 | 10/5 | 100/99/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526656` | 09-27 09:30 | 2 | greedy test (ranking-1790525401-seed1) | exp-base | greedy; 9 to ±64 | 10/5 | 100/99/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526519` | 09-27 09:28 | 2 | greedy test (ranking-1790525401-seed1) | ppo-k8@500 | greedy; 9 to ±64 | 10/5 | 100/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526381` | 09-27 09:26 | 2 | greedy test (ranking-1790525401-seed1) | ppo-wm3 | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526243` | 09-27 09:24 | 2 | greedy test (ranking-1790525401-seed1) | ppo-wm3-lr1e3-5k | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790526105` | 09-27 09:21 | 2 | greedy test (ranking-1790525401-seed1) | exp-upright25 | greedy; 9 to ±64 | 10/5 | 99/99/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525968` | 09-27 09:19 | 2 | greedy test (ranking-1790525401-seed1) | exp-ent03 | greedy; 9 to ±64 | 10/5 | 100/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525830` | 09-27 09:17 | 2 | greedy test (ranking-1790525401-seed1) | ppo-k8@1000 | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525692` | 09-27 09:14 | 2 | greedy test (ranking-1790525401-seed1) | ppo-speed | greedy; 9 to ±64 | 10/5 | 99/99/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525554` | 09-27 09:12 | 2 | greedy test (ranking-1790525401-seed1) | ppo-k8 | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525416` | 09-27 09:10 | 2 | greedy test (ranking-1790525401-seed1) | exp-lr1e3-2k | greedy; 9 to ±64 | 10/5 | 100/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525278` | 09-27 09:08 | 2 | greedy test (ranking-1790523610-seed0) | ppo-wm3-lr1e3-5k | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525140` | 09-27 09:05 | 2 | greedy test (ranking-1790523610-seed0) | ppo-k8 | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790525003` | 09-27 09:03 | 2 | greedy test (ranking-1790523610-seed0) | ppo-speed | greedy; 9 to ±64 | 10/5 | 100/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524865` | 09-27 09:01 | 2 | greedy test (ranking-1790523610-seed0) | exp-base | greedy; 9 to ±64 | 10/5 | 99/99/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524727` | 09-27 08:58 | 2 | greedy test (ranking-1790523610-seed0) | exp-lr1e3-2k | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524589` | 09-27 08:56 | 2 | greedy test (ranking-1790523610-seed0) | ppo-wm3 | greedy; 9 to ±64 | 10/5 | 99/97/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524452` | 09-27 08:54 | 2 | greedy test (ranking-1790523610-seed0) | exp-upright25 | greedy; 9 to ±64 | 10/5 | 99/99/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524314` | 09-27 08:51 | 2 | greedy test (ranking-1790523610-seed0) | ppo-wm2 | greedy; 9 to ±64 | 10/5 | 100/99/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524176` | 09-27 08:49 | 2 | greedy test (ranking-1790523610-seed0) | ppo-k8@1000 | greedy; 9 to ±64 | 10/5 | 99/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790524038` | 09-27 08:47 | 2 | greedy test (ranking-1790523610-seed0) | exp-ent03 | greedy; 9 to ±64 | 10/5 | 99/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790523901` | 09-27 08:45 | 2 | greedy test (ranking-1790523610-seed0) | ppo-k8@250 | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790523763` | 09-27 08:42 | 2 | greedy test (ranking-1790523610-seed0) | ppo-k8@500 | greedy; 9 to ±64 | 10/5 | 99/98/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |
| `1790523625` | 09-27 08:40 | 2 | greedy test (ranking-1790523610-seed0) | ppo-k8@100 | greedy; 9 to ±64 | 10/5 | 99/99/99 | train: wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5 |

### 2026-09-26

2 recordings, 2.1 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790464876` | 09-26 16:21 | 120 | collection | ppo-wm3 | sampled; 9 to ±64 | 10/5 | 99/97/100 | train: wm:base, wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790464558` | 09-26 16:16 | 5 | collection | ppo-wm3 | sampled; 9 to ±64 | 10/5 | 99/97/100 | val_policy_wm3: wm:base, wm:random_walk_only, wm:stochastic-2, wm:stochastic-3, wm:stochastic; val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |

### 2026-09-25

4 recordings, 2.7 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790378922` | 09-25 16:28 | 120 | collection | ppo-wm2 | sampled; 9 to ±64 | 10/5 | 99/98/99 | train: wm:base, wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:base, policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790378604` | 09-25 16:23 | 5 | collection | ppo-wm2 | sampled; 9 to ±64 | 10/5 | 100/98/100 | val_policy_wm2: wm:base, wm:random_walk_only; val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790374602` | 09-25 15:16 | 5 | collection | ppo-speed | greedy; 9 to ±64 | 10/5 | 99/98/100 | val_policy: wm:base, wm:random_walk_only; val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic |
| `1790372784` | 09-25 14:46 | 30 | collection | ppo-speed | greedy; 9 to ±64 | 10/5 | 99/98/100 | train: wm:base, wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:base, policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |

### 2026-09-24

3 recordings, 0.8 hours.

| recording | start | min | kind | policy | actions | drive/rest s | tags seen % | used in |
|---|---|---|---|---|---|---|---|---|
| `1790282998` | 09-24 13:49 | 25 | random walk | random walk | random walk; ±60, steps ≤ 40 | 20/5 | 97/96/100 | train: wm:base, wm:random_walk_only, wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:base, policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790281442` | 09-24 13:24 | 10 | random walk | random walk | random walk; ±60, steps ≤ 40 | 20/5 | 97/96/99 | val_random_walk: wm:base, wm:random_walk_only, wm:stochastic-2, wm:stochastic-3, wm:stochastic; val_all: wm:stochastic-2, wm:stochastic-3, wm:stochastic; starts: policy:base, policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
| `1790279628` | 09-24 12:53 | 10 | random walk | random walk | random walk; ±60, steps ≤ 40 | 20/5 | 97/96/99 | train: wm:base, wm:random_walk_only, wm:stochastic-2, wm:stochastic-3, wm:stochastic, stoch-it1, stoch-it2, stoch-it3, stoch-it4, stoch-it5; starts: policy:base, policy:wm-k1, policy:wm-k6, policy:wm-k8, policy:wm-s256-anneal, policy:wm-stoch, ppo-s256, ppo-s256-anneal, ppo-s256-anneal0-5k, ppo-s256-anneal0-8k, ppo-s256-mean, ppo-stoch-it1, ppo-stoch-it2, ppo-stoch-it3, ppo-stoch-it4 |
