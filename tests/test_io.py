# tests/test_io.py

import pandas as pd
import pandas.testing as pdt
import pytest
import json

from pyHSP.io import (
    read_hsdx,
    read_hsdx_result,
    to_json,
    hsdx_to_json,
    hsdx_to_csv,
)

# ---------------------------------------------------------------------------
# read_hsdx
# ---------------------------------------------------------------------------


# @pytest.mark.parametrize(
#     "filename",
#     [
#         "HSDX.hsdx",
#         "HSDX_DA.hsdx",
#     ],
# )
def test_read_hsdx(data_dir):
    result = read_hsdx(data_dir / "HSDX.hsdx")

    expected = pd.DataFrame(
        {
            "dD": [14.5, 16.0],
            "dP": [8.0, 11.7],
            "dH": [13.5, 10.2],
            "MVol": [57.6, 95.0],
            "CAS": ["64-19-7", "108-24-7"],
            "SMILES": ["CC(O)=O", "CC(OC(C)=O)=O"],
            "Score": [1.0, 1.0],
            "dH_donor": [11.5, 0.0],
            "dH_acceptor": [7.1, 10.2],
        },
        index=[
            "SolventA",
            "SolventB",
        ],
    )
    expected.index.name = "Name"

    pdt.assert_frame_equal(result, expected)
    assert result.index.name == "Name"


# ---------------------------------------------------------------------------
# read_hsdx_result
# ---------------------------------------------------------------------------

HSDX = pd.Series([16.8, 9.3, 4.2, 10.1], index=["dD", "dP", "dH", "radius"])
HSDX_DA = pd.Series(
    [17.7, 9.5, 0.0, 4.8, 9.7],
    index=["dD", "dP", "dH_donor", "dH_acceptor", "radius"],
)


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("HSDX.hsdx", HSDX),
        ("HSDX_DA.hsdx", HSDX_DA),
    ],
)
def test_read_hsdx_result(filename, expected, data_dir, extended):
    # make sure we don't mutate the series for later tests
    expected = expected.copy()

    result = read_hsdx_result(data_dir / filename, extended=extended)
    if extended:
        expected["In"] = 62
        expected["Out"] = 26
        expected["Wrong In"] = 4
        expected["Wrong Out"] = 8
        expected["Fit"] = 0.838
    pdt.assert_series_equal(result, expected)


def test_read_hsdx_no_result(data_dir):
    with pytest.raises(ValueError, match="Error matching pattern"):
        read_hsdx_result(data_dir / "no_result.hsdx")


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------


def test_to_json_uses_index_orientation():
    df = pd.DataFrame(
        {
            "dD": [18.2],
            "dP": [5.6],
            "dH": [7.4],
        },
        index=["Example"],
    )

    result = json.loads(to_json(df))

    assert result == {
        "Example": {
            "dD": 18.2,
            "dP": 5.6,
            "dH": 7.4,
        }
    }


# ---------------------------------------------------------------------------
# File Conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        (None, "HSDX.json"),
        ("test.json", "test.json"),
    ],
)
def test_hsdx_to_json_creates_file(out, expected, tmp_path, data_dir):
    # copy file to tmp
    infile = tmp_path / "HSDX.hsdx"
    infile.write_bytes((data_dir / "HSDX.hsdx").read_bytes())

    if out:
        out = tmp_path / out
    assert hsdx_to_json(infile, out) is None
    assert (tmp_path / expected).exists()


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        (None, "HSDX.csv"),
        ("test.csv", "test.csv"),
    ],
)
def test_hsdx_to_csv_creates_file(out, expected, tmp_path, data_dir):
    # copy file to tmp
    infile = tmp_path / "HSDX.hsdx"
    infile.write_bytes((data_dir / "HSDX.hsdx").read_bytes())

    if out:
        out = tmp_path / out
    assert hsdx_to_csv(infile, out) is None
    assert (tmp_path / expected).exists()
