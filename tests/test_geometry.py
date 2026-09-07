"""Tests for `sttopt.geometry` and `sttopt.geometry_builders`."""

import numpy as np
import pytest

import sttopt.geometry as geometry
import sttopt.geometry_builders as geometry_builders


def _write_npz(tmp_path, **arrays):
    path = tmp_path / "geometry.npz"
    np.savez(path, **arrays)
    return path


# --- load_geometry ---------------------------------------------------------------


def test_load_geometry_round_trips(tmp_path):
    xPhys = np.zeros((5, 7))
    xPhys[:, :3] = 1.0
    path = _write_npz(tmp_path, xPhys=xPhys)
    np.testing.assert_array_equal(geometry.load_geometry(path), xPhys)


def test_load_geometry_missing_key(tmp_path):
    path = _write_npz(tmp_path, other=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="other"):
        geometry.load_geometry(path)


def test_load_geometry_rejects_3d(tmp_path):
    path = _write_npz(tmp_path, xPhys=np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="2-D"):
        geometry.load_geometry(path)


def test_load_geometry_rejects_out_of_range(tmp_path):
    xPhys = np.zeros((3, 4))
    xPhys[0, 0] = 1.5
    path = _write_npz(tmp_path, xPhys=xPhys)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        geometry.load_geometry(path)


def test_load_geometry_rejects_non_binary(tmp_path):
    xPhys = np.zeros((3, 4))
    xPhys[0, 0] = 0.5
    path = _write_npz(tmp_path, xPhys=xPhys)
    with pytest.raises(ValueError, match="not binary"):
        geometry.load_geometry(path)


def test_load_geometry_non_binary_waived(tmp_path):
    xPhys = np.full((3, 4), 0.5)
    path = _write_npz(tmp_path, xPhys=xPhys)
    np.testing.assert_array_equal(geometry.load_geometry(path, binary=False), xPhys)


# --- base_elements -----------------------------------------------------------------


def test_base_elements_overhang_bracket_bottom_row():
    """The bottom row of the overhang bracket is solid only left of the 50mm line --
    the test that would catch "whole bottom edge" reasoning."""
    nelx = nely = 20  # square raster of the square 100x100mm bracket
    xPhys = geometry_builders.overhang_bracket(nelx, nely)

    bottom_row = (nely - 1) * nelx + np.arange(nelx)
    got = geometry.base_elements(xPhys, bottom_row)
    got_cols = np.sort(got % nelx)

    # The left half of the bottom row sits on the build plate; the right half is under
    # the 45-degree cut and does not. "Whole bottom edge" reasoning would return every
    # column in bottom_row instead of this proper subset.
    assert 0 < got_cols.size < nelx
    assert got_cols[0] == 0  # leftmost column: solid
    assert got_cols[-1] < nelx - 1  # rightmost column: not included
    np.testing.assert_array_equal(
        got_cols, np.arange(got_cols.size)
    )  # a contiguous run


def test_base_elements_raises_when_lifted_clear():
    xPhys = np.zeros((5, 5))
    xPhys[0, :] = 1.0  # solid only at the top, away from the "bottom row" candidates
    bottom_row = 4 * 5 + np.arange(5)
    with pytest.raises(ValueError, match="build plate"):
        geometry.base_elements(xPhys, bottom_row)


# --- geometry_builders ---------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_name,corner_solid",
    [
        ("overhang_bracket", None),
        ("c_shape", None),
        ("l_shape", None),
    ],
)
def test_builder_nonsquare_mesh_runs(shape_name, corner_solid):
    """Every builder must work on a non-square mesh -- conventions.md: a square
    fixture can pass a transposed implementation undetected."""
    builder, width, height = geometry_builders.SHAPES[shape_name]
    nelx, nely = 23, 17
    xPhys = builder(nelx, nely, width=width, height=height)
    assert xPhys.shape == (nely, nelx)
    assert set(np.unique(xPhys)) <= {0.0, 1.0}
    assert 0.0 < xPhys.mean() < 1.0


def test_overhang_bracket_solid_fraction_and_corners():
    nelx = nely = 40
    xPhys = geometry_builders.overhang_bracket(nelx, nely)
    # Area = 50*50 (left block) + 50*100 (right column) - 50*50/2 (cut triangle)
    # = 2500 + 5000 - 1250 = 6250, out of 10000 total -> fraction 0.625.
    assert xPhys.mean() == pytest.approx(0.625, abs=0.02)
    assert xPhys[-1, 0] == 1.0  # bottom-left corner: solid
    assert xPhys[0, -1] == 1.0  # top-right corner: solid (right column, full height)
    assert xPhys[0, 0] == 0.0  # top-left corner: void (only the lower half is a block)


def test_c_shape_solid_fraction_and_slot():
    nelx, nely = 40, 48  # matches the 200x240 aspect ratio
    xPhys = geometry_builders.c_shape(nelx, nely)
    # Area = 200*240 - (200-60)*(180-80) = 48000 - 14000 = 34000 -> fraction 34/48.
    assert xPhys.mean() == pytest.approx(34000 / 48000, abs=0.02)
    dx, dy = 200.0 / nelx, 240.0 / nely
    slot_row = nely - 1 - int(130 / dy)  # y=130mm mid-slot -> row from the top
    slot_col = int(150 / dx)  # x=150mm, inside the slot
    assert xPhys[slot_row, slot_col] == 0.0


def test_l_shape_solid_fraction_and_corners():
    nelx = nely = 40  # 200x200 is square
    xPhys = geometry_builders.l_shape(nelx, nely)
    # Area = 200*200 - (200-80)*(200-80) = 40000 - 14400 = 25600 -> fraction 0.64.
    assert xPhys.mean() == pytest.approx(0.64, abs=0.02)
    assert xPhys[-1, 0] == 1.0  # bottom-left: solid (base)
    assert xPhys[-1, -1] == 1.0  # bottom-right: solid (base)
    assert xPhys[0, 0] == 1.0  # top-left: solid (vertical arm)
    assert xPhys[0, -1] == 0.0  # top-right: void


def test_binarize_reports_solid_fraction(tmp_path, capsys):
    xPhys = np.array([[0.1, 0.4], [0.6, 0.9]])
    src = _write_npz(tmp_path, xPhys=xPhys)
    out = tmp_path / "binarized.npz"
    args = geometry_builders._parse_args(
        ["binarize", str(src), "--threshold", "0.5", "--out", str(out)]
    )
    geometry_builders.main(args)

    with np.load(out) as data:
        np.testing.assert_array_equal(data["xPhys"], [[0.0, 0.0], [1.0, 1.0]])
        assert float(data["threshold"]) == 0.5


# --- neighbor_pairs / drop_disconnected ---------------------------------------------


def test_neighbor_pairs_are_symmetric_with_the_right_step_lengths():
    src, dst, step = geometry.neighbor_pairs(3, 3)

    # The centre element of a 3x3 grid has all 8 neighbours; the corners have 3.
    assert int(np.count_nonzero(src == 4)) == 8
    assert int(np.count_nonzero(src == 0)) == 3
    # Every adjacency appears in both directions, with the same step length.
    both = set(zip(src.tolist(), dst.tolist()))
    assert both == {(b, a) for a, b in both}
    assert set(np.round(step, 6)) == {1.0, round(np.sqrt(2), 6)}


def test_drop_disconnected_removes_an_island_and_warns():
    xPhys = np.zeros((5, 6))
    xPhys[-1, :] = 1.0  # a base row on the build plate
    xPhys[1, 3] = 1.0  # an island floating clear of it
    base = geometry.base_elements(xPhys, np.arange(6) + 4 * 6)

    with pytest.warns(UserWarning, match="no path of material"):
        cleaned = geometry.drop_disconnected(xPhys, base)

    assert cleaned[1, 3] == 0.0
    np.testing.assert_array_equal(cleaned[-1, :], 1.0)
    np.testing.assert_array_equal(xPhys[1, 3], 1.0)  # the input is not mutated


def test_drop_disconnected_keeps_material_joined_only_diagonally():
    """8-connected, matching the step lengths the geodesic initialization uses -- a
    corner-touching element is connected, not an island."""
    xPhys = np.zeros((4, 4))
    xPhys[-1, 0] = 1.0
    xPhys[-2, 1] = 1.0  # touches the base element at a corner only
    base = geometry.base_elements(xPhys, np.arange(4) + 3 * 4)

    cleaned = geometry.drop_disconnected(xPhys, base)

    np.testing.assert_array_equal(cleaned, xPhys)


def test_drop_disconnected_returns_the_input_when_nothing_is_disconnected():
    xPhys = np.zeros((3, 3))
    xPhys[-1, :] = 1.0
    base = geometry.base_elements(xPhys, np.arange(3) + 2 * 3)

    assert geometry.drop_disconnected(xPhys, base) is xPhys
