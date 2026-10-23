from __future__ import annotations


from collections.abc import Callable, Mapping


import pytest


from kiln.ir.plan import (
    DEFAULT_ALIGNMENT,
    Lifetime,
    OpStep,
    TensorInfo,
    allocate_arena,
    calculate_liveness,
)


def tensors(**sizes: int) -> dict[str, TensorInfo]:

    return {name: TensorInfo(name, size) for name, size in sizes.items()}


def lifetimes(**spans: tuple[int, int]) -> dict[str, Lifetime]:

    return {name: Lifetime(*span) for name, span in spans.items()}


def region(offsets: Mapping[str, int], name: str, size: int) -> range:

    start = offsets[name]

    return range(start, start + size)


def are_disjoint(left: range, right: range) -> bool:

    return left.stop <= right.start or right.stop <= left.start


@pytest.fixture
def chain() -> list[OpStep]:

    return [
        OpStep(0, ["input"], ["score"]),
        OpStep(1, ["score"], ["aux"]),
        OpStep(2, ["aux"], ["result"]),
    ]


class TestCalculateLiveness:
    def test_graph_output_lifetime_is_preserved(
        self,
        chain: list[OpStep],
    ) -> None:

        liveness = calculate_liveness(
            chain,
            graph_outputs=["score"],
        )

        assert liveness == lifetimes(
            input=(0, 0),
            score=(0, 2),
            aux=(1, 2),
            result=(2, 2),
        )

    def test_input_tensor_starts_at_first_scheduled_op(self) -> None:

        liveness = calculate_liveness(
            [OpStep(7, ["x"], ["y"])],
        )

        assert liveness["x"] == Lifetime(7, 7)

    def test_schedule_is_normalized_by_op_id(self) -> None:

        ops = [
            OpStep(2, ["b"], ["c"]),
            OpStep(0, ["x"], ["a"]),
            OpStep(1, ["a"], ["b"]),
        ]

        liveness = calculate_liveness(ops)

        assert liveness == lifetimes(
            x=(0, 0),
            a=(0, 1),
            b=(1, 2),
            c=(2, 2),
        )

    def test_empty_schedule_has_no_lifetimes(self) -> None:

        assert calculate_liveness([]) == {}


class TestAllocateArena:
    def test_graph_output_cannot_alias_a_live_tensor(
        self,
        chain: list[OpStep],
    ) -> None:

        liveness = calculate_liveness(
            chain,
            graph_outputs=["score"],
        )

        plan = allocate_arena(
            tensors(
                input=64,
                score=64,
                aux=64,
                result=64,
            ),
            liveness,
        )

        score = region(plan.offsets, "score", 64)

        for name in ("aux", "result"):
            assert are_disjoint(
                score,
                region(plan.offsets, name, 64),
            ), name

    def test_graph_output_cannot_alias_input(
        self,
        chain: list[OpStep],
    ) -> None:

        liveness = calculate_liveness(
            chain,
            graph_outputs=["score"],
        )

        plan = allocate_arena(
            tensors(
                input=64,
                score=64,
                aux=64,
                result=64,
            ),
            liveness,
        )

        assert are_disjoint(
            region(plan.offsets, "input", 64),
            region(plan.offsets, "score", 64),
        )

    def test_disjoint_lifetimes_reuse_the_same_region(self) -> None:

        plan = allocate_arena(
            tensors(a=64, b=64),
            lifetimes(
                a=(0, 1),
                b=(2, 3),
            ),
        )

        assert plan.offsets["a"] == plan.offsets["b"]

        assert plan.total_arena_bytes == 64

    def test_overlapping_lifetimes_use_distinct_regions(self) -> None:

        plan = allocate_arena(
            tensors(a=64, b=64),
            lifetimes(
                a=(0, 2),
                b=(2, 3),
            ),
        )

        assert plan.offsets["a"] != plan.offsets["b"]

        assert plan.total_arena_bytes == 128

    def test_first_fit_reuses_gaps_before_extending_arena(self) -> None:

        plan = allocate_arena(
            tensors(a=64, b=128, c=64),
            lifetimes(
                a=(0, 0),
                b=(0, 2),
                c=(1, 3),
            ),
        )

        assert dict(plan.offsets) == {
            "a": 0,
            "b": 64,
            "c": 0,
        }

        assert plan.total_arena_bytes == 192

    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (0, 0),
            (1, 4),
            (4, 4),
            (5, 8),
            (7, 8),
            (8, 8),
            (9, 12),
        ],
    )
    def test_sizes_are_rounded_up_to_the_alignment(
        self,
        size: int,
        expected: int,
    ) -> None:

        plan = allocate_arena(
            tensors(x=size),
            lifetimes(x=(0, 0)),
            alignment=DEFAULT_ALIGNMENT,
        )

        assert plan.total_arena_bytes == expected

    def test_peak_concurrent_bytes_excludes_reusable_memory(self) -> None:

        plan = allocate_arena(
            tensors(
                a=64,
                b=64,
                c=64,
            ),
            lifetimes(
                a=(0, 0),
                b=(1, 1),
                c=(2, 2),
            ),
        )

        assert plan.total_arena_bytes == 64

        assert plan.peak_concurrent_bytes == 64

    def test_peak_concurrent_bytes_counts_overlapping_buffers(self) -> None:

        plan = allocate_arena(
            tensors(
                a=64,
                b=128,
            ),
            lifetimes(
                a=(0, 2),
                b=(1, 3),
            ),
        )

        assert plan.total_arena_bytes == 192

        assert plan.peak_concurrent_bytes == 192

    def test_offsets_are_immutable(self) -> None:

        plan = allocate_arena(
            tensors(x=64),
            lifetimes(x=(0, 0)),
        )

        with pytest.raises(TypeError):
            plan.offsets["x"] = 128

    def test_empty_graph_produces_an_empty_plan(self) -> None:

        plan = allocate_arena({}, {})

        assert plan.total_arena_bytes == 0

        assert plan.peak_concurrent_bytes == 0

        assert dict(plan.offsets) == {}


class TestValidation:
    @pytest.mark.parametrize(
        ("call", "expected_error", "match"),
        [
            pytest.param(
                lambda: calculate_liveness(
                    [OpStep(0, ["x"], ["y"])],
                    graph_outputs=["missing"],
                ),
                KeyError,
                "unknown graph output",
                id="unknown-graph-output",
            ),
            pytest.param(
                lambda: calculate_liveness(
                    [
                        OpStep(0, ["x"], ["y"]),
                        OpStep(0, ["y"], ["z"]),
                    ],
                ),
                ValueError,
                "duplicate op_id",
                id="duplicate-op-id",
            ),
            pytest.param(
                lambda: calculate_liveness(
                    [
                        OpStep(0, ["x"], ["y"]),
                        OpStep(1, ["y"], ["y"]),
                    ],
                ),
                ValueError,
                "multiple definitions",
                id="multiple-definitions",
            ),
            pytest.param(
                lambda: allocate_arena(
                    tensors(x=64),
                    lifetimes(missing=(0, 0)),
                ),
                KeyError,
                "unknown tensor",
                id="unknown-tensor",
            ),
            pytest.param(
                lambda: allocate_arena(
                    tensors(x=64),
                    lifetimes(x=(2, 1)),
                ),
                ValueError,
                "invalid lifetime",
                id="invalid-lifetime",
            ),
            pytest.param(
                lambda: allocate_arena(
                    tensors(x=-1),
                    lifetimes(x=(0, 0)),
                ),
                ValueError,
                "negative tensor size",
                id="negative-tensor-size",
            ),
            pytest.param(
                lambda: allocate_arena(
                    tensors(x=64),
                    lifetimes(x=(0, 0)),
                    alignment=0,
                ),
                ValueError,
                "invalid alignment",
                id="invalid-alignment",
            ),
        ],
    )
    def test_invalid_input_is_rejected(
        self,
        call: Callable[[], object],
        expected_error: type[Exception],
        match: str,
    ) -> None:

        with pytest.raises(expected_error, match=match):
            call()
