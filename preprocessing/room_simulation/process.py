from numpy.typing import ArrayLike
import time
import random
import pyroomacoustics as pra
import numpy as np


def sample_rt60(rng: random.Random) -> float:
    """Sample a plausible RT60 in seconds.

    Keep it in a reasonable range to avoid pathological absorption values.
    """

    # Typical small/medium room reverberation times.
    return rng.uniform(0.4, 1.0)


def sample_room_dim(rng: random.Random) -> list[float]:
    """Sample room dimensions [x, y, z] in meters."""

    # Keep z (height) smaller than x/y; enforce minimums to avoid near-degenerate rooms.
    x = rng.uniform(5.0, 15.0)
    y = rng.uniform(5.0, 15.0)
    z = rng.uniform(2, 6)
    return [x, y, z]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sample_position_in_room(
    rng: random.Random,
    room_dim: list[float],
    *,
    wall_margin: float,
    z_range: tuple[float, float] | None = None,
) -> list[float]:
    """Sample a position [x,y,z] inside the shoebox while keeping distance from walls."""

    x_max, y_max, z_max = room_dim

    # Ensure the margin can't exceed half the dimension.
    mx = min(wall_margin, x_max / 2.0 - 1e-6)
    my = min(wall_margin, y_max / 2.0 - 1e-6)
    mz = min(wall_margin, z_max / 2.0 - 1e-6)

    if z_range is None:
        z_lo, z_hi = mz, z_max - mz
    else:
        # Also respect room bounds and wall margin.
        z_lo = _clamp(z_range[0], mz, z_max - mz)
        z_hi = _clamp(z_range[1], mz, z_max - mz)
        if z_hi < z_lo:
            z_lo, z_hi = z_hi, z_lo

    return [
        rng.uniform(mx, x_max - mx),
        rng.uniform(my, y_max - my),
        rng.uniform(z_lo, z_hi),
    ]


def sample_source_and_mic_positions(
    rng: random.Random,
    room_dim: list[float],
    *,
    wall_margin: float = 0.5,
    min_distance: float = 1.0,
    max_tries: int = 200,
) -> tuple[list[float], list[float]]:
    """Sample (source_pos, mic_pos) with constraints.

    Constraints:
    - Both positions lie within the room bounds with a wall margin.
    - The positions are at least `min_distance` apart.
    """

    # Reasonable height ranges for typical scenarios.
    source_z = (1.0, min(2.0, room_dim[2] - wall_margin))
    mic_z = (1.0, min(1.8, room_dim[2] - wall_margin))

    for _ in range(max_tries):
        s = sample_position_in_room(
            rng, room_dim, wall_margin=wall_margin, z_range=source_z
        )
        m = sample_position_in_room(
            rng, room_dim, wall_margin=wall_margin, z_range=mic_z
        )
        if float(np.linalg.norm(np.array(s) - np.array(m))) >= min_distance:
            return s, m

    raise RuntimeError(
        f"Could not sample valid source/mic positions after {max_tries} tries "
        f"(room_dim={room_dim}, wall_margin={wall_margin}, min_distance={min_distance})."
    )


def simulate_room(sample: ArrayLike, samplerate: int) -> tuple[np.ndarray, pra.ShoeBox]:
    t0 = time.perf_counter()

    # Randomize acoustic parameters for each run.
    # NOTE: uses the module-level seed above for reproducibility.
    rt60 = sample_rt60(random)  # seconds
    room_dim = sample_room_dim(random)  # meters
    print(f"  rt60={rt60:.3f}s, room_dim={[round(d, 3) for d in room_dim]}")

    source_pos, mic_pos = sample_source_and_mic_positions(
        random,
        room_dim,
        wall_margin=0.6,
        min_distance=1.2,
    )
    print(
        "  positions: "
        f"source={[round(v, 3) for v in source_pos]}, "
        f"mic={[round(v, 3) for v in mic_pos]}"
    )

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=samplerate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    room.add_source(source_pos, signal=sample, delay=1.3)
    mic_locs = np.c_[mic_pos]
    room.add_microphone_array(mic_locs)
    room.simulate()

    dt = time.perf_counter() - t0
    print(f"  processing took {dt:.3f}s")

    return room.mic_array.signals[0], room
