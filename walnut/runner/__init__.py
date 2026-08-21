"""Execution: what it takes to turn a batch of rows into a batch of tokens.

The device-facing half of the serving loop. Everything here holds tensors,
knows what a model is, and is specific to the hardware the model runs on —
which is exactly what the half above it (`walnut.scheduler`) must not be.

`walnut.runner.graphs` captures the decode step per batch bucket, so a step
that is bound by launch overhead pays it once at start-up rather than every
token. `walnut.runner.sampling` draws the tokens themselves, grouping the rows
of a batch that asked for the same thing into one call.

The direction of the dependency is worth stating, because it is easy to get
backwards: the runner imports the scheduler's *vocabulary* — a request's
sampling parameters, and in time the plan for a step — and the scheduler
imports nothing from here. A plan is data. Running it is not.
"""

from walnut.runner.graphs import DecodeGraph, DecodeGraphs, buckets
from walnut.runner.sampling import Sampler

__all__ = ["DecodeGraph", "DecodeGraphs", "Sampler", "buckets"]
