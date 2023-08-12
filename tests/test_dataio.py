import pandas as pd
import pytest

from uv_monitor.dataio import parse_header, read_csv_file


@pytest.fixture
def sample_csv(tmp_path):
    data = """#Location: Arizona,Flagstaff
#Latitude: 36.059000 N
#Longitude: 112.184000 W
#Elevation: 2073.000 m
#Internal ID: AZ01
Column1,Column2,Column3
1,2,3
4,5,6
"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    return file


def test_parse_header():
    header = """#Location: Arizona,Flagstaff
#Latitude: 36.059000 N
#Longitude: 112.184000 W
#Elevation: 2073.000 m
#Internal ID: AZ01"""
    expected_output = {
        "loc_name": "Arizona,Flagstaff",
        "lat": 36.059,
        "lon": -112.184,
        "elev": 2073.0,
        "loc_id": "AZ01",
    }
    assert parse_header(header) == expected_output


def test_read_csv_file(sample_csv):
    expected_df = pd.DataFrame(
        {"Column1": [1, 4], "Column2": [2, 5], "Column3": [3, 6]}
    )
    expected_header = """#Location: Arizona,Flagstaff
#Latitude: 36.059000 N
#Longitude: 112.184000 W
#Elevation: 2073.000 m
#Internal ID: AZ01\n"""
    df, header = read_csv_file(sample_csv)
    assert df.equals(expected_df)
    assert header == expected_header
