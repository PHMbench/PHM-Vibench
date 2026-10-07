from pathlib import Path

import pandas as pd

from data.utils.update_002_XJTU_metadata import update_xjtu_sy_metadata_v3_staging


def test_xjtu_conditions_remain_distinct_without_mutating_input(tmp_path: Path) -> None:
    conditions = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    rows = [
        {
            "Name": "RM_002_XJTU",
            "File": f"{condition}/Bearing{index + 1}_1/1.csv",
            "Domain_id": -1,
            "Domain_description": "unassigned",
        }
        for index, condition in enumerate(conditions)
    ]
    rows.append(
        {
            "Name": "OTHER",
            "File": "other/acquisition.csv",
            "Domain_id": 17,
            "Domain_description": "preserve existing condition",
        }
    )
    source = pd.DataFrame(rows).assign(
        Sample_rate=1000,
        Sample_lenth=256,
        Channel=1,
        Label=7,
        RUL_label=0.5,
        Fault_Diagnosis=False,
        Anomaly_Detection=False,
        Remaining_Life=False,
        Digital_Twin_Prediction=False,
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    source.to_csv(input_path, index=False)
    original_bytes = input_path.read_bytes()

    returned = update_xjtu_sy_metadata_v3_staging(str(input_path), str(output_path))
    persisted = pd.read_csv(output_path)

    for actual in (returned, persisted):
        xjtu = actual.loc[actual["Name"].eq("RM_002_XJTU")]
        assert xjtu["Domain_id"].tolist() == [0, 1, 2]
        assert xjtu["Domain_description"].tolist() == conditions
        pd.testing.assert_frame_equal(
            actual.loc[actual["Name"].eq("OTHER")],
            source.loc[source["Name"].eq("OTHER")],
            check_dtype=False,
        )
    assert input_path.read_bytes() == original_bytes
