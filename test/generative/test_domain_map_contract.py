import pandas as pd
import pytest

from src.data_factory.ID.domain_map import hash_file, load_domain_map, validate_domain_map


def test_dummy_domain_map_loads_and_hashes():
    path = "configs/domain_maps/dummy_domain_map.csv"
    df = load_domain_map(path)

    assert list(df["domain_id"]) == [0, 1]
    assert len(hash_file(path)) == 64


def test_domain_map_rejects_duplicate_domain_ids():
    df = pd.DataFrame(
        {
            "domain_id": [0, 0],
            "system_id": [0, 0],
            "load": ["a", "b"],
            "rpm": ["1000", "1200"],
            "sampling_rate": [1000, 1000],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        validate_domain_map(df)

