"""Net verification of rod builds against an injected run.

A structure's own rod blueprint (``rod_topology_from_deconstruction``)
is a P1 cell with one node slot per chemical repeat and no edge
centers: run detection finds nothing on it, and its cut list never
held the rod's own continuation. Verification must therefore be told
the run and supply the continuation from it - the corpus rod
self-template arm recorded zero verifications out of 203
composition-exact rebuilds before it did.
"""

from __future__ import annotations

import copy

import pytest

from .test_deconstruct import FIXTURE_PATH, _rod_pillar_structure


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def pillar_self(mofgen):
    """(deconstruction, topology, run, laterals, rod fragment) of the pillar."""
    from autografs.extract_topology import rod_topology_from_deconstruction

    result = mofgen.deconstruct(_rod_pillar_structure(1))
    topology, run, lateral_mapping, fragment = rod_topology_from_deconstruction(result)
    laterals = {
        index: copy.deepcopy(result.fragments[name])
        for index, name in lateral_mapping.items()
    }
    return result, topology, run, laterals, fragment


def _rebuild(pillar_self, **overrides):
    from autografs.rod_build import build_rod_framework

    _result, topology, run, laterals, fragment = pillar_self
    kwargs = dict(
        run=run,
        min_distance=None,
        bond_tolerance=10.0,
        initial_scale=1.0,
        scale_band=0.25,
    )
    kwargs.update(overrides)
    return build_rod_framework(topology, fragment, laterals, **kwargs)


class TestSelfRunContinuation:
    def test_single_repeat_run_gets_a_self_loop_on_its_generator(self, pillar_self):
        # the blueprint's cut list holds only rod-lateral bonds; the
        # rod-form quotient must add the continuation the build realizes
        from autografs.net import topology_quotient_edges, topology_rod_quotient_edges

        _result, topology, run, _laterals, _fragment = pillar_self
        (node,) = run.nodes
        raw = topology_quotient_edges(topology)
        assert not any(a == b == node for a, b, _ in raw)
        poe = topology_rod_quotient_edges(topology, run)
        loops = [v for a, b, v in poe if a == b == node]
        assert len(loops) == 1
        assert loops[0] in (tuple(run.direction), tuple(-x for x in run.direction))
        # laterals untouched: everything else is the raw quotient
        assert sum(poe.values()) == sum(raw.values()) + 1

    def test_library_run_is_untouched(self, mofgen):
        # a detected run carries edge centers, so contraction already
        # yields the continuation and nothing is synthesised on top
        from dataclasses import replace

        from autografs.net import axial_runs, topology_rod_quotient_edges

        pcu = mofgen.topologies["pcu"]
        run = axial_runs(pcu)[0]
        node = next(
            s for s in run.slots if len(pcu.slots[s].atoms.indices_from_symbol("X")) > 2
        )
        declared = replace(run, nodes=(node,))
        assert topology_rod_quotient_edges(
            pcu, declared
        ) == topology_rod_quotient_edges(pcu, run)


class TestVerifyAgainstInjectedRun:
    def test_self_rebuild_verifies_only_when_told_the_run(self, pillar_self):
        from autografs.exceptions import NetMismatchError

        _result, topology, run, _laterals, _fragment = pillar_self
        built = _rebuild(pillar_self)
        # detection has nothing to read on a P1 self-blueprint ...
        with pytest.raises(NetMismatchError):
            built.verify_net(topology)
        # ... the injected run is the blueprint the build answers to
        built.verify_net(topology, runs=[run])

    def test_build_time_gate_uses_the_caller_run(self, pillar_self):
        _rebuild(pillar_self, verify_net=True)

    def test_dropped_rod_lateral_bond_fails(self, pillar_self):
        from autografs.exceptions import NetMismatchError
        from autografs.framework import Framework

        _result, topology, run, _laterals, _fragment = pillar_self
        built = _rebuild(pillar_self)
        graph = built.graph.copy()
        u, v = next(
            (a, b)
            for a, b in graph.edges()
            if graph.nodes[a]["sbu"] != graph.nodes[b]["sbu"]
        )
        graph.remove_edge(u, v)
        with pytest.raises(NetMismatchError):
            Framework(graph, name="miswired").verify_net(topology, runs=[run])

    def test_run_on_the_wrong_slot_fails(self, pillar_self):
        # the run must name the rod's own slots: declaring a lateral as
        # the run puts the continuation on the wrong vertex and the
        # signatures part. (A merely re-based generator is NOT a
        # different net - a, b, a+c generate the same pcu - so that is
        # deliberately not what is tested here.)
        from dataclasses import replace

        from autografs.exceptions import NetMismatchError

        _result, topology, run, laterals, _fragment = pillar_self
        built = _rebuild(pillar_self)
        lateral = next(iter(laterals))
        misplaced = replace(run, slots=(lateral,), nodes=(lateral,))
        with pytest.raises(NetMismatchError):
            built.verify_net(topology, runs=[misplaced])
