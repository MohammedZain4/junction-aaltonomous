# FastSort: a dual-arm, instruction-aware sorter for Väiski

FastSort is our entry to the Aaltonomous Humanoid challenge. It is a classical, fully autonomous policy:
- **perception:** head RGB-D plus the wrist cameras
- **planning:** inverse kinematics with a full-arm collision model
- **instructions:** a robust parser for free-form text

It ships as one CPU-only Docker policy server (`aaltonomous/docker/fastsort.Dockerfile`) and needs no training and
no GPU. Real-robot data was used only to tune and check the perception.

## What it does differently

| Idea | Why it matters |
|---|---|
| **Both arms at once.** Each arm has its own queue of moves. Before any move is committed, the complete future paths of both arms (palm, fingers, fingertips, wrist, elbow) are checked against each other, and one arm is delayed only if they would come within 7.5 cm. | About twice the throughput of one arm at a time. Episodes typically finish in 20–60 s of the 120 s budget. |
| **Chaining.** An arm goes straight from one drop to its next pick, without returning home. Planning runs in a background thread, so the robot never waits for the planner between picks. | No dead time between picks. |
| **Free-form instructions.** Any phrasing, synonyms, typos, "respectively", "the rest", "except", "only", negation-free rules, "my left", and Finnish, Swedish and German. The real colours map automatically (orange → red, purple → blue), and "orange box" is understood as a bin. | The finals' heaviest episode (×1.5) is an unseen instruction. Tested on all 288 official template permutations plus 50+ unseen paraphrases. |
| **Shape-aware grasping.** The shape (cube, cylinder or triangular prism) comes from the 3D top face. Cubes are pinched across parallel faces, triangles parallel to a side, and a retry starts from the next grasp axis. Each object's shape is a confidence-weighted vote across looks. | Fewer slips; shape-based instructions work. |
| **Self-calibration from the bins.** At the first look, the bin walls (fixed and known) are fitted in the depth image, and the camera's xy error is corrected. | Object position error fell from 12.5 to 7.2 mm mean (13.9 to 5.3 median) under camera jitter. |
| **Colours tuned on real footage.** Colour is decided per blob, not per pixel, against reference hues from the sim and from the real head camera. Objects touching a bin are split from it. The robot's own arm and hands are excluded. | On the 49 Drop 2 frames, every purple, orange and yellow object is labelled correctly, and false purple detections fell from 12 to about 2 per frame. |
| **Wrist-camera checks.** After each lift, the wrist camera checks the hand really holds the object, aborting and retrying at once if not, and double-checks the colour, putting the object back and re-sorting it if the head camera got it wrong. The checks are mount-agnostic: they rely on the held object moving with the camera. | The sim mounts the wrist cameras on the forearm, the real robot in the palm (Drop 2), so anything tuned to one view fails on the other. |
| **Honest timing.** `AALTO_RT_EMULATE=1` charges all planning and vision wall time to the sim clock. Without it, a sim running at 0.15× real time hides about 85% of the thinking time that the real robot will pay for. | Our scores reflect real-time conditions. |

## Results (sim, our L40S box)

| Test | Result |
|---|---|
| Safety gate (0 clipped steps, ≥ 27 Hz command rate, own latency ≥ 27 Hz) | Passed on every run: 0 clips, 46–82 Hz own rate |
| Fixed sort, nominal scene | ~10/12 objects |
| Instruction sort, nominal scene | 10/10 |
| Fixed sort, domain-randomised (textures, lighting, camera jitter), seeds 1000–1002, after the scheduling fixes | 3/3, 5/5 (then a relocation-loop fix for the third scene) |
| Same, real-time emulated | 3/3, 4/5, 2/4 |
| Perception, 12 DR scenes | 54/54 objects found, 0 phantoms, shape accuracy 47/54 |

## Findings worth knowing

- **The lid stretch goal is out of reach with the published layout.** With legs and waist locked, the left hand cannot get over any hole: it falls short by 2.6 cm (triangle), 6.5 cm (cylinder) and 9.3 cm (square). This holds even with the hand at any angle (checked with the URDF kinematics). The right arm is 13–22 cm short.
- **Cross-body carries never work.** A left arm carrying to the right bin, or the reverse, failed in all 26 tries across 7 evaluation runs. FastSort hands such objects over instead: one arm puts the object down within the other arm's reach.
- **The sim and real wrist cameras are mounted differently** (forearm vs palm), as noted above.
- **The sim runs at 0.15× real time on a 2 GHz EPYC.** Physics is CPU-bound and the GPU sits idle, which is why we built the real-time emulation.

## Files

| File | What |
|---|---|
| `policy.py` | Dual-arm lanes, background planner thread, collision delay search, look / choose / commit, grip check actions, real-time emulation |
| `grasp.py` | Shape and colour perception (`ShapePerception`) and the shape-aware grasp planner (`ShapePlanner`) |
| `instruction.py` | Instruction parser → `Rules` (colour and shape → bin) |
| `calib.py` | Head-camera xy self-calibration from the bin walls |
| `wristcheck.py` | Grip-check features from the wrist camera and finger joints |
| `policy_server.py` | Websocket policy server (the submission entry point) |
| `perception_bench.py`, `grasp_bench.py` | Benchmarks against sim ground truth (perception confusion, one-shot grasps, labelled grip features, `--true-xy`) |
| `test_instruction.py`, `test_grasp.py` | Unit tests (stdlib + numpy / OpenCV) |

## Run

```bash
docker build -f aaltonomous/docker/fastsort.Dockerfile -t fastsort .     # from the repo root
docker run --rm -p 8000:8000 fastsort                                    # policy server on :8000
aalto eval --policy ws://localhost:8000 --task sort_language             # in the sim container
```

Environment switches (all optional):
- `AALTO_GRIP_CHECK=log|act|0`
- `AALTO_RT_EMULATE=1`
- `AALTO_SELF_CALIB=0`
- `AALTO_SHAPE_GRASP=0`
