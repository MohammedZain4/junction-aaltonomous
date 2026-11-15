# Junction submission kit (hackjunction.app)

Deadline: **Sun 2026-11-15 10:00**. Partners judge the digital materials; peer review is at the on-site demo exhibition.

| Field | What we hand in | Status |
|---|---|---|
| Demo (digital) | Sim evaluation clips + the Docker policy server (`docker/fastsort.Dockerfile`, image `fastsort:v8`, built from commit 1508c1a; v7 = previous fallback) | v8 built; sim validation 3/3 + 4/5 so far (v7: 2/3 + 4/5) |
| Source code | GitHub repo (make it **public** before submitting) | private now |
| Video (max 2 min) | Demo reel v2 (`scripts/make_demo.sh`, 38 s, `fastsort_demo_v2.mp4`) | to review together |
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
  German. It was tested on all 288 official templates plus 50+ unseen paraphrases. A shape-insertion ("through the hole in
  the lid") instruction is recognised and the objects are left in place rather than dropped in the wrong bins.
- **Shape-aware grasping.** The head camera's depth image gives each object's top face and shape, as probabilities
  (e.g. "80% triangle") that are combined over several looks before a grasp is chosen. Hard cases are handled
  explicitly: prisms lying on their side (the ridge on top), tilted cubes that show only half their top face, and
  tilted cylinders that show only a sliver. On fresh randomised scenes shapes are right 93% of the time (57/61), and
  cubes are no longer mistaken for prisms (from about 45% before). The grasp axis is chosen per shape.
- **Checks itself.** At the first look, it calibrates the camera against the known bin walls, which cut position
  error by 40%. After every lift, a grip check uses the finger joints, the wrist camera and the head camera to
  confirm the object is in the hand; if the hand is empty, the arm aborts at once and the pick is retried.
- **Built for the real robot.** Colour thresholds were tuned on real head-camera footage (orange, purple and yellow
  objects; orange bins). The sim scene was recoloured to match the real objects and bins. It works within the 120 s
  limit and through the organisers' safety filter. From the real-robot depth recordings we found the robot leans
  about 14° towards the table; at the first look FastSort fits the table plane and, when it sees such a lean, plans
  both the camera view and the arm motions in a level table frame. Without that, 10 of 13 grasps planned on real
  frames would have pushed the hand into the table; with it, none do.

Tech: Python, NumPy, inverse kinematics with a full-arm collision model, RGB-D perception, Isaac Sim for evaluation.
