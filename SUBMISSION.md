# Junction submission kit (hackjunction.app)

Deadline: **Sun 2026-11-15 10:00**. Partners judge the digital materials; peer review is at the on-site demo exhibition.

| Field | What we hand in | Status |
|---|---|---|
| Demo (digital) | Sim evaluation clips + the Docker policy server (`docker/fastsort.Dockerfile`, image `fastsort:v5`) | rebuild v5 after the final fixes |
| Source code | GitHub repo (make it **public** before submitting) | private now |
| Video (max 2 min) | Demo reel (`/root/make_demo.sh`, draft 53 s) | redo: the draft "wasn't great" |
| Team info | Solo entry, filled in by hand on the platform | to do (you) |
| Project description | The text below | draft ready |

## Project description (paste into the platform)

**FastSort: a dual-arm, instruction-aware sorter for the Unitree G1 "Väiski"**

FastSort is a fully autonomous policy that makes the G1 humanoid sort coloured cubes, cylinders and triangular
prisms into three bins, either by the fixed rule (yellow → left, purple → centre, orange → right) or by a
free-form spoken-style instruction. It ships as a single CPU-only Docker policy server: no training, no GPU.

- **Both arms at once.** Each arm has its own queue of moves. Every move is checked against the other arm's whole
  future path (palm, fingers, wrist, elbow) before it is committed, so the two arms work in parallel without
  colliding. That gives about twice the throughput of a one-arm sorter.
- **Understands instructions.** It handles any phrasing, including synonyms, typos, "the rest", "except",
  "respectively" and "my left". It follows rules by colour, by shape or by both, in English, Finnish, Swedish and
  German. It was tested on all 288 official templates plus 50+ unseen paraphrases.
- **Shape-aware grasping.** The head camera's depth image gives each object's top face and shape. Prisms lying on
  their side are still recognised, from the ridge on top. The grasp axis is chosen per shape.
- **Checks itself.** At the first look, it calibrates the camera against the known bin walls, which cut position
  error by 40%. After every lift, a grip check uses the finger joints, the wrist camera and the head camera to
  confirm the object is in the hand; if the hand is empty, the arm aborts at once and the pick is retried.
- **Built for the real robot.** Colour thresholds were tuned on real head-camera footage (orange, purple and yellow
  objects; orange bins). The sim scene was recoloured to match the real objects and bins. It works within the 120 s
  limit and through the organisers' safety filter.

Tech: Python, NumPy, inverse kinematics with a full-arm collision model, RGB-D perception, Isaac Sim for evaluation.
