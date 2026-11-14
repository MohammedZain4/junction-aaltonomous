"""FastSort behind the submission protocol (port 8000).

    python -m aaltonomous.fastsort.policy_server
    docker build -f aaltonomous/docker/fastsort.Dockerfile -t fastsort:v1 .

Env: AALTO_TIME_LIMIT (s, default 120), AALTO_PROFILE (baseline config profile, default `sim`),
AALTO_FAST=0 (baseline motion speeds instead of the fast overrides).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .policy import FastSortPolicy

log = logging.getLogger("aaltonomous.fastsort")


class FastSortServerPolicy(Policy):
    name = "fastsort"

    def __init__(self, time_limit=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        self.pol = FastSortPolicy(time_limit=tl, log=lambda m: log.info(m), **kw)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])

    def reset(self, instruction, task=None, seed=None):
        self.pol.reset({"instruction": instruction})
        log.info("reset task=%s seed=%s", task, seed)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = FastSortServerPolicy()
    log.info("fastsort loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
