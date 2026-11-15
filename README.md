# Aaltonomous Humanoid — Junction 2026

> ### For judges: submitted image and eval
> - **Docker image:** `ghcr.io/mohammedzain4/fastsort:v8` (public)
> - **Image digest:** `sha256:c41f65a5d25e3179865f6397ecb18e41fcadee6010c5e555225bdaf061a731e7`
> - **Run:** `docker run --rm -p 8000:8000 ghcr.io/mohammedzain4/fastsort:v8` (CPU only), then evaluate against `ws://localhost:8000`
> - **aalto eval output** (this image, sort_fixed):
>
> ```
>   task / variant             sort_fixed / train
>   trials                     1
>   score mean ± std           +3.00 ± 0.00  (100 % of max)
>   objects correct            3/3 (100 %)
>   episodes all correct       1/1
>   outcomes                   {'correct': 3}
>   end reasons                {'done': 1}
>   safety clips total / max   0 / 0
>   min tick / command rate    30.0 / 29.97 Hz (sim clock)
>   min policy real-time rate  63.0 Hz
>   sim real-time factor       mean 0.135  min 0.135
>   SIM GATE                   PASS
> ```
>
> Full validation (new code, Isaac Sim, randomisation on): fixed rule 10/12 objects, instructions 9/13, 0 safety clips.

> **Our entry: [FastSort](aaltonomous/fastsort/README.md).** A dual-arm, instruction-aware sorter: both arms
> working at once with full-arm collision checks, free-form multilingual instructions, shape-aware grasps,
> self-calibration from the bins, colour detection tuned on the real Drop 2 footage, and wrist-camera grip checks.
> The rest of this README is the organisers' starter-kit documentation.

Starter code for the **Aaltonomous Humanoid** challenge: make Väiski, a Unitree G1 with two Dex3 hands, sort
coloured objects into bins fully autonomously. Develop in sim, hand in a Docker policy server, the best teams run on
the real robot in the finals.

| Start here | |
|---|---|
| [Challenge brief](aaltonomous/docs/participants/README.md) | Task, rungs, timeline, what to read next |
| [Quickstart](QUICKSTART.md) | Fresh cloud GPU instance → baseline running in sim (~15 min) |
| [Rules and scoring](aaltonomous/docs/participants/RULES_AND_SCORING.md) | Success rule, points, finalist selection |
| [Interface](aaltonomous/docs/participants/INTERFACE.md) | Observations, actions, joint order, limits, cameras |
| [Submission](aaltonomous/docs/participants/SUBMISSION.md) | Docker policy-server contract (websocket + msgpack, port 8000) |
| [Starter paths](aaltonomous/docs/participants/STARTER_PATHS.md) | Classical, VLM, IL/VLA, RL, real2sim |
| [FAQ](aaltonomous/docs/participants/FAQ.md) | |

## Layout

| Path | What |
|---|---|
| `aaltonomous/interface/` | `AaltoEnv`, joint spec, the safety filter every action goes through |
| `aaltonomous/deploy/`, `aaltonomous/docker/` | Policy-server template and submission Dockerfiles |
| `aaltonomous/sim/` | Isaac Lab task `Aalto-Sort-G129-Dex3-Joint`, evaluator, oracle, training DR config |
| `aaltonomous/scene/` | Scene geometry (`scene.yaml`), success checker, printable assets |
| `aaltonomous/baselines/classical/` | Classical perception + IK baseline |
| `aaltonomous/data/` | Real-demo recorder and LeRobot / GR00T converters |
| `aaltonomous/cloud/` | Instance setup, `aalto` CLI, sim image Dockerfile |

Unit tests (no GPU needed): `python -m unittest aaltonomous/interface/test_safety.py aaltonomous/interface/test_env.py
aaltonomous/scene/test_scene.py`.

Upstream dependencies (`unitree_sim_isaaclab`, `xr_teleoperate` for the G1 URDF, Isaac Lab) are cloned into
`third_party/` by `aaltonomous/cloud/setup_instance.sh`; the sim Docker image already contains them.

Organisers: DigiTwin, Aalto University. This repo is synced from our internal repo; please open issues or ask in the
challenge channel rather than sending PRs during the event.
