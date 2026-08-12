from types import SimpleNamespace

import pandas as pd

from src.data_factory.ID.Id_searcher import search_ids_for_task
from src.data_factory.data_utils import MetadataAccessor


def test_default_task_passes_complete_id_universe_to_registered_split(capsys) -> None:
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {"Id": 17, "Dataset_id": 2, "Label": 0},
                {"Id": 21, "Dataset_id": 2, "Label": 1},
                {"Id": 34, "Dataset_id": 2, "Label": 1},
            ]
        ),
        key_column="Id",
    )

    train_val, test = search_ids_for_task(
        metadata,
        SimpleNamespace(target_system_id=[2], type="Default_task"),
    )

    assert train_val == [17, 21, 34]
    assert test == [17, 21, 34]
    assert "not specifically handled" not in capsys.readouterr().out
