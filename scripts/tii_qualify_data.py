"""Read-only qualification of the 18 supplied local TII HDF5 containers.

Run from the PHMFactory repository with ``python -m scripts.tii_qualify_data``.
This does not train, select frequency bands from signals, or repair metadata.
The evidence notes below refer to supplied acquisition documents, not model results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.data_factory.data_utils import read_metadata_table
from src.task_factory.Components.tii_target_head import FixedEpisodeError, make_episode
from src.utils.identifiers import validate_identifiers


FILES = (
    "RM_001_CWRU", "RM_002_XJTU", "RM_003_FEMTO", "RM_004_IMS",
    "RM_005_Ottawa23", "RM_006_THU", "RM_007_MFPT", "RM_008_UNSW",
    "RM_010_SEU", "RM_015_susu", "RM_016_JNU", "RM_017_Ottawa19",
    "RM_018_THU24", "RM_020_DIRG", "RM_023_HIT23", "RM_024_JUST",
    "RM_027_PU", "RM_031_HUST24",
)

# Documentary observations are deliberately explicit. A sensor's catalogue range
# is not automatically the response of the sensor/conditioner/ADC acquisition.
EVIDENCE: dict[str, dict[str, str]] = {
    "RM_001_CWRU": dict(
        group="unknown physical bearing identifier; file numbers and load Domain_id are not independent groups",
        unit="original acquisition calibration/output scale not documented sufficiently; do not assume g",
        native="12000 and 48000 Hz; original benchmark Table A1 identifies baseline97-100 as48000 while metadata says12000",
        sensor="drive-end and fan-end acceleration; current reader retains DE_time/FE_time",
        processing="reader truncates DE/FE to shorter length then concatenates; no resampling/filtering/normalization in reader; original site mentions unspecified MATLAB processing",
        support="acquisition transfer function, passband tolerance and transition guards not established",
        license="collection Apache-2.0 defers to original dataset terms; original explicit license not established",
        evidence="raw/RM_001_CWRU; src/data_factory/reader/RM_001_CWRU.py; https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures",
        blockers="physical bearing groups unresolved; raw/channel sampling provenance unresolved; acquisition support unresolved"),
    "RM_002_XJTU": dict(
        group="physical bearing: Bearing1_1..Bearing3_5 parsed from original path; 15 accelerated-life experiments",
        unit="raw CSV unit not declared in header; acceleration figures/source calibration require unit confirmation",
        native="25600 Hz (local original paper section 1.2)",
        sensor="PCB 352C33 horizontal and vertical on bearing housing; DT9837 DAQ",
        processing="reader reads both CSV signal columns; no resampling/filtering/normalization",
        support="sensor model and DAQ known; actual response/tolerance/guards not established for cached traces",
        license="publicly released for research in original paper; explicit redistribution license not found",
        evidence="raw/RM_002_XJTU/XJTU-SY滚动轴承加速寿命试验数据集解读.pdf sections 1.2-1.3/table 3; metadata labels",
        blockers="labels include -1 degradation and one fault label per bearing rather than validated fault types; physical unit unresolved; acquisition support unresolved"),
    "RM_003_FEMTO": dict(
        group="physical bearing from Bearing1_1..Bearing3_3 path; 17 run-to-failure bearings",
        unit="unknown from supplied CSV and metadata; temperature and vibration mixed",
        native="vibration nominal 25600 Hz; temperature has separate acquisition semantics not captured in metadata",
        sensor="acc_ columns 4/5 vibration; temp_ is temperature, not a vibration channel",
        processing="reader selects columns 4:6; Bearing1_4 special semicolon parsing; zero-channel temperature records survive old cache",
        support="no acquisition response/tolerance/guards found in supplied dataset documents",
        license="collection Apache-2.0; original FEMTO challenge terms not supplied",
        evidence="raw/RM_003_FEMTO/{Learning_set,Test_Set}/Bearing*/{acc_,temp_}*.csv; native reader; all H5 shapes",
        blockers="mixed temperature/vibration population; labels include missing/-1 and bearing-index fault codes; physical unit and acquisition support unresolved"),
    "RM_004_IMS": dict(
        group="independent experiment run directory: 1st_test/2nd_test/4th_test; each record simultaneously measures multiple bearings",
        unit="unknown from local timestamped text records and metadata",
        native="20000 Hz reported in metadata and local README; no raw timestamps within sampled record",
        sensor="8 channels first run; 4 channels later runs; bearing/channel fault attribution not encoded by row labels",
        processing="tab-delimited values retained; no resampling/filtering/normalization in reader",
        support="no supplied acquisition response/tolerance/guard evidence",
        license="collection Apache-2.0; original IMS license not established locally",
        evidence="raw/RM_004_IMS/{1st_test,2nd_test,4th_test}; metadata Label_Description; native reader",
        blockers="labels encode run/health stage, not channel-resolved fault type; fault classes confined to one run each; physical unit and acquisition support unresolved"),
    "RM_005_Ottawa23": dict(
        group="bearing number in H/I/O/B/C_<bearing>_<trial>; same bearing across healthy and faulty records",
        unit="acceleration m/s^2; source paper conversion applied; raw reader selects columns 0,1,4",
        native="42000 Hz in original acquisition paper and metadata; local README 100000 Hz is inconsistent",
        sensor="PCB 623C01 magnet-mounted drive-end; 482C-series conditioner; NI USB-6212 ADC; cache channels vibration m/s^2, acoustic V, temperature degrees C",
        processing="reader drops CSV header then selects columns0/1/4 vibration/acoustic/temperature; no resampling/filtering/normalization",
        support="PCB623C01 sensor: 2.4-8000 Hz +/-5%; 482C exact model/filter settings absent; NI USB6212 has no anti-alias filter; end-to-end support unknown",
        license="CC BY 4.0 original dataset https://data.mendeley.com/datasets/y2px5tg92h/2",
        evidence="raw/RM_005_Ottawa23/1-s2.0-S2352340923004456-main.pdf sections2.1/3.4-3.6; https://www.pcb.com/products?m=623c01; https://forums.ni.com/t5/Multifunction-DAQ/Does-the-NI-USB-6212-Mulitfunction-USB-DAQ-have-input-anti/td-p/723566",
        blockers="conditioner transfer/filter settings and acquisition transition guards unresolved"),
    "RM_006_THU": dict(
        group="unknown physical bearing/run IDs; health directory and speed are not independent groups",
        unit="vibration and energy-harvester voltage units/calibration unresolved",
        native="20480 Hz reported by local metadata/README; original acquisition needs confirmation",
        sensor="vibration and piezoelectric harvester voltage",
        processing="raw vibration MAT has horizontal/vertical acceleration, exported voltage has different rate; H5 second-channel transformation unresolved; sampled native-reader mismatch",
        support="effective acquisition passband/tolerance/guards unknown",
        license="non-open subset: local README requires contact with original authors",
        evidence="raw/RM_006_THU; local README THU section; native reader",
        blockers="raw/cache transformation unresolved; physical groups and physical scale unresolved; acquisition support unresolved; original permission not established"),
    "RM_007_MFPT": dict(
        group="bearing/run identity across load trials not documented sufficiently; fault folders are not independent groups",
        unit="raw field gs suggests g but explicit calibration/unit evidence not established",
        native="raw MAT sr: lab97656/48828Hz; OilPump24414Hz and Planet6104Hz contradict metadata48828Hz",
        sensor="single vibration channel from bearing MAT gs",
        processing="cache equals raw bearing.gs for all23 records; native reader fails missing critical BPFO field for lab files; raw frequencies not resampled",
        support="sensor/conditioner/anti-alias frequency response not established",
        license="original explicit license not found; old MFPT data URL now redirects; collection Apache-2.0 cannot establish original data terms",
        evidence="raw/RM_007_MFPT original documentation/MAT sr/gs; RAW_ACQUISITION_CHECKS.csv; native reader; old mfpt.org data URL now redirects to ASNT",
        blockers="physical groups and physical unit unresolved; fourth class combines three real-world machines; two metadata sample rates wrong; acquisition support unresolved"),
    "RM_008_UNSW": dict(
        group="Test 1..Test 4 are original run-to-failure experiments; speed folders are conditions within each experiment",
        unit="accH/accV in V; 10 mV/(m/s^2), so multiply acceleration channels by 100; other channels have different units",
        native="51200 Hz metadata and MAT Fs; local README 6 Hz is rotation rate, not sampling rate",
        sensor="horizontal/vertical accelerometers plus encoder/load/speed; original channel names in MAT/docx",
        processing="MAT reader concatenates channels; no resampling/filtering/normalization",
        support="acceleration sensitivity documented; effective acquisition passband/tolerance/guards unresolved",
        license="original distribution license not established locally",
        evidence="raw/RM_008_UNSW/Read me for data description.docx and MAT accH/accV/Fs; native reader",
        blockers="labels are test-specific defect/load progression rather than established local fault classes; one original run per label; acquisition support unresolved"),
    "RM_010_SEU": dict(
        group="unknown bearing/gear specimen IDs; speed/load conditions and file names are not independent physical groups",
        unit="unknown; channels mix vibration and torque",
        native="5120 Hz in local metadata/README; original time calibration not established",
        sensor="8 channels: motor/gearbox vibration and motor torque",
        processing="reader drops CSV header lines and index/trailing column; retained 8 channels; no resampling/filtering/normalization",
        support="original CSV header Frequency Limit=2000Hz; filter response/tolerance/guards and calibrated physical units unknown",
        license="public source code/data links; original data license not established locally",
        evidence="raw/RM_010_SEU/gearbox/{bearingset,gearset}; native reader; metadata Dataset_id 9 and 15 in one H5",
        blockers="physical groups and unit unresolved; acquisition support unresolved"),
    "RM_015_susu": dict(
        group="original README describes five bearings/health states; repeated speeds are not new bearings; explicit specimen IDs absent",
        unit="unknown physical calibration in supplied MAT/README",
        native="31175 Hz raw source README/metadata; collection README 25600 Hz inconsistent",
        sensor="South Ural State University source according to original README; collection origin claims inconsistent",
        processing="MAT signal columns retained by reader; no declared resampling/filtering/normalization",
        support="acquisition response/tolerance/guards unknown",
        license="original explicit license not established locally",
        evidence="raw/RM_015_susu original README; metadata; native reader",
        blockers="specimen IDs and physical scale unresolved; acquisition support unresolved"),
    "RM_016_JNU": dict(
        group="unknown physical bearing IDs; three rotation rates per fault type do not establish independent groups",
        unit="unknown calibration in original CSV/metadata",
        native="50000 Hz metadata/local README; raw time axis unavailable",
        sensor="single vertical acceleration channel on bearing",
        processing="reader loads numeric CSV; no resampling/filtering/normalization",
        support="acquisition response/tolerance/guards unknown",
        license="original explicit license not established locally",
        evidence="raw/RM_016_JNU; local README JNU section; native reader",
        blockers="physical groups and physical scale unresolved; acquisition support unresolved"),
    "RM_017_Ottawa19": dict(
        group="bearing identities not enumerated; filenames encode class/speed profile/repetition, not proven independent specimens",
        unit="original vibration acceleration conversion needs confirmation; encoder is a separate modality",
        native="200000 Hz original paper and metadata; local README 42000 Hz describes another corpus",
        sensor="Channel_1 vibration; Channel_2 encoder; original acquisition paper",
        processing="reader concatenates Channel_1 and Channel_2 without resampling/filtering/normalization",
        support="effective sensor/conditioner/ADC passband and guards not established",
        license="original dataset terms require source verification; collection Apache-2.0 is not sufficient",
        evidence="raw/RM_017_Ottawa19/Bearing vibration data collected under time-varying.pdf sections2.1-2.2; H-A-1.mat; native reader",
        blockers="physical bearing groups and physical scale unresolved; acquisition support unresolved"),
    "RM_018_THU24": dict(
        group="unknown physical specimen identities; trials within health-state folders may reuse specimen",
        unit="both exported channels V (CSV Vertical Units); conversion to common acceleration unavailable",
        native="native before filtering unknown; exported sample interval319.99999192us is approximately3125Hz",
        sensor="smart squirrel cage sensing; channels cannot be assumed equivalent acceleration",
        processing="raw CSV says Filtered CH1/Filtered CH2; unknown pre-export filters; reader extracts columns4/10 and pandas header omits first signal sample",
        support="acquisition response/tolerance/guards unknown",
        license="non-open subset: local README requires contact with original authors",
        evidence="raw/RM_018_THU24; local README THU24 section; native reader",
        blockers="physical groups and physical scale unresolved; acquisition support unresolved; original permission not established"),
    "RM_020_DIRG": dict(
        group="original specimen codes 0A..6A; C4A and E4A share bearing 4A (changing endurance filenames are not new groups)",
        unit="acceleration m/s^2 documented in original paper",
        native="51200 Hz radial C* recordings; 102400 Hz endurance E* recordings (original paper; verify metadata consistency)",
        sensor="two triaxial accelerometers; OR38 acquisition; sensor range 1-12000 Hz +/-5%",
        processing="reader extracts six-channel MAT array, float64; no resampling/filtering/normalization",
        support="sensor1-12000Hz +/-5%; OR38 manufacturer anti-alias>400dB/oct, bandwidth0.45Fs, ripple+/-0.005dB. Conservative in-band support has evidence; >12kHz is unknown rather than certified unavailable",
        license="open dataset in original paper; explicit license must follow archive source terms",
        evidence="data/Reference/RM_020_DIRG/Daga 等 - 2019 - The Politecnico di Torino rolling bearing test rig.pdf sections2.2-3.2; https://wiki.oros.com/index.php?title=Hardware_Specification",
        blockers="normal class has only one physical bearing, preventing disjoint source train/validation; unavailable incremental atoms above documented response not established"),
    "RM_023_HIT23": dict(
        group="unknown specimen IDs; fault/severity/speed names do not establish separate bearings",
        unit="unknown physical calibration in MAT/metadata",
        native="51200 Hz metadata/original source; collection README 25600 Hz inconsistent",
        sensor="single vibration channel; locally self-built source",
        processing="reader extracts MAT signal; no declared resampling/filtering/normalization",
        support="acquisition response/tolerance/guards unknown",
        license="original explicit license not established locally",
        evidence="raw/RM_023_HIT23/Self-built dataset; metadata descriptions; native reader",
        blockers="physical groups and physical scale unresolved; acquisition support unresolved"),
    "RM_024_JUST": dict(
        group="B1/I1/O1/H1 state identifiers may reuse specimens; conditions/trials are not proven independent groups",
        unit="raw CSV explicitly six vibration channels m/s^2 and one AE channel dB; AE cannot be treated as acceleration",
        native="50000Hz original paper and raw time increment20us; collection README25600Hz inconsistent",
        sensor="Kistler8702B100/8763B100BB acceleration and8152C0050511 AE; DEWESOFTX3; H5 seven channels, metadata three",
        processing="reader retains raw numeric columns; channel selection/units not resolved",
        support="acquisition response/tolerance/guards unknown",
        license="original explicit license not established locally",
        evidence="data/Reference/RM_024_JUST/Study of JUST Slewing Bearing Failure Test Data.pdf sections1.2-1.3; raw CSV headers; native reader; H5 vs metadata shapes",
        blockers="channel and length contract inconsistent; physical specimen groups unresolved; acquisition support unresolved"),
    "RM_027_PU": dict(
        group="32 physical bearing codes from first File directory; operating conditions and trials remain within bearing",
        unit="raw MAT Y Unit fields empty, including vibration_1; paper sensor model alone does not fix cache numerical scale",
        native="64000Hz nominal original paper; Id47567 raw256823 points/4.00000342s yields64205.445Hz average, metadata256001 points; inspect RAW_ACQUISITION_CHECKS.csv",
        sensor="channel0/1 motor currents; channel2 vibration_1 (PCB336C04); Kistler5015A low-pass30kHz",
        processing="reader selects Y current_1/current_2/vibration_1 and concatenates; no resampling/filtering/normalization",
        support="30 kHz analogue low-pass documented; sensor/conditioner tolerance and transition guards incomplete for binary support",
        license="CC BY-NC4.0 original data https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter",
        evidence="data/Reference/RM_027_PU/PHME16_CM_bearing.pdf and K001/measuring_log_K001.pdf; raw MAT Y.Name/Y.Unit/X; native reader",
        blockers="physical scale of cached vibration unresolved; raw time-axis uniformity/rate conflict; complete acquisition response/tolerance/guards unresolved"),
    "RM_031_HUST24": dict(
        group="original author README describes 9 bearing health states x 11 speeds; no independent per-speed bearing IDs",
        unit="acquisition screenshot says g but Volts/unit=1 versus sensor100mV/g; scaling conflict unresolved",
        native="25600 Hz original README/metadata",
        sensor="TREA331 three-axis; original sensor photo0.5-10000Hz +/-3dB, sensitivity100mV/g +/-15%",
        processing="reader parses source spreadsheets; no declared resampling/filtering/normalization",
        support="raw Frequency Limit10000Hz and sensor0.5-10000Hz +/-3dB documented; actual filter transition/unavailable response unknown",
        license="author Readme permits diagnostic-algorithm validation and requests citation; no standard license name",
        evidence="data/Reference/RM_031_HUST24/Description file（描述文件）/Readme for HUSTbearing dataset.pdf; raw Description file（描述文件）/F4-sensor type.png and F5-signal collection setting.jpg; native reader",
        blockers="independent bearing groups and physical scale unresolved; acquisition support unresolved"),
}


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def physical_group(name: str, raw_file: str) -> str | None:
    """Only documented specimen/run identities, never Domain_id or row Id."""
    if name in {"RM_002_XJTU", "RM_003_FEMTO"}:
        found = re.search(r"Bearing\d+_\d+", raw_file)
        return found.group() if found else None
    if name in {"RM_004_IMS", "RM_008_UNSW", "RM_027_PU"}:
        return raw_file.split("/")[0]
    if name == "RM_020_DIRG":
        found = re.match(r"[CE](\d+A)", Path(raw_file).name)
        return found.group(1) if found else None
    if name == "RM_005_Ottawa23":
        found = re.match(r"[HIOBC]_(\d+)_\d+\.csv", Path(raw_file).name)
        return f"bearing_{int(found.group(1))}" if found else None
    return None


def fixed_episode(rows: list[dict[str, object]]) -> dict[str, object]:
    """Describe the one prescribed metadata-only draw, including failed counts.

    This qualification computation does not produce a training split or a model.
    It retains counts from failed draws, which the runtime rightly rejects.
    """
    result: dict[str, object] = dict(episode_seed=1729, episode_status="NOT_RUN",
                                   support_record_count="unknown", support_group_count="unknown",
                                   query_group_count="unknown", episode_reason="")
    labels = [r["label"] for r in rows]
    valid_counts = Counter(int(x) for x in labels if not pd.isna(x) and float(x).is_integer() and float(x) >= 0)
    if valid_counts and min(valid_counts.values()) < 5:
        result.update(episode_status="FAIL", episode_reason="fewer than five original recordings in a declared class; no draw/redraw")
        return result
    if any(pd.isna(x) or not float(x).is_integer() or float(x) < 0 for x in labels):
        result["episode_reason"] = "invalid/missing/negative fault labels; no records dropped or relabelled"
        return result
    if any(r["group"] is None for r in rows):
        result["episode_reason"] = "physical group provenance missing; no surrogate group constructed"
        return result
    episode_input = [{"recording_id": int(r["recording_id"]), "group": r["group"], "label": int(r["label"])} for r in rows]
    try:
        episode = make_episode(episode_input, shots=5, seed=1729)
        result["episode_status"] = "PASS"
    except FixedEpisodeError as exc:
        episode = exc.episode
        result.update(episode_status="FAIL", episode_reason=str(exc))
    except ValueError as exc:
        result.update(episode_status="FAIL", episode_reason=str(exc))
        return result
    classes = sorted({r["label"] for r in episode})
    support = [r for r in episode if r["role"] == "support"]
    query = [r for r in episode if r["role"] == "query"]
    support_records = {str(c): sum(int(r["label"]) == c for r in support) for c in classes}
    support_count = {str(c): len({r["group"] for r in support if int(r["label"]) == c}) for c in classes}
    query_count = {str(c): len({r["group"] for r in query if int(r["label"]) == c}) for c in classes}
    result.update(support_record_count=compact(support_records), support_group_count=compact(support_count),
                  query_group_count=compact(query_count))
    return result


def run(root: Path, output: Path) -> None:
    started = time.monotonic()
    root = root.resolve()
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("derived qualification outputs must be outside the immutable data root")
    output.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata_table(root / "metadata.xlsx")
    validate_identifiers(metadata["Id"], "recording_id")
    if metadata["Id"].duplicated().any():
        raise ValueError("metadata Id is not unique")
    records: list[dict[str, object]] = []
    qualifications: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    acquisition: list[dict[str, object]] = []
    selected_metadata = metadata.loc[metadata.Name.isin(FILES)]
    duplicate_paths = selected_metadata.duplicated(["Name", "File"], keep=False)
    duplicate_ids = set(selected_metadata.loc[duplicate_paths, "Id"])
    # Resolve symlinks as well: two nominal datasets must not treat the same
    # underlying local recording as independent observations.
    resolved_raw = selected_metadata.apply(lambda row: str((root / "raw" / row.Name / str(row.File)).resolve()), axis=1)
    shared_raw_ids = set(selected_metadata.loc[resolved_raw.duplicated(keep=False), "Id"])
    inode_ids: dict[tuple[int, int], list[int]] = {}
    for row in selected_metadata.itertuples(index=False):
        path = root / "raw" / row.Name / row.File
        if path.is_file():
            stat = path.stat()
            inode_ids.setdefault((stat.st_dev, stat.st_ino), []).append(int(row.Id))
    shared_inode_ids = {key for ids in inode_ids.values() if len(ids) > 1 for key in ids}

    for name in FILES:
        subset = metadata.loc[metadata.Name == name].copy()
        if subset.empty:
            raise ValueError(f"no metadata rows for declared file {name}.h5")
        validate_identifiers(subset["File"], "raw recording path")
        note = EVIDENCE[name]
        reader = importlib.import_module(f"src.data_factory.reader.{name}").read
        local_records: list[dict[str, object]] = []
        with h5py.File(root / f"{name}.h5", "r") as h5:
            expected = {str(int(x)) for x in subset.Id}
            keys = set(h5.keys())
            missing, extra = sorted(expected - keys), sorted(keys - expected)
            attrs_count = 0
            for key in keys:
                data = h5[key]
                attrs_count += bool(data.attrs)
            for key in extra:
                anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="H5 key has no current metadata row", observation=str(h5[key].shape)))
            probe_categories: set[tuple[object, ...]] = set()
            for row in subset.itertuples(index=False):
                key = str(int(row.Id))
                raw_path = root / "raw" / name / row.File
                data = h5[key] if key in h5 else None
                shape = tuple(data.shape) if data is not None else ()
                group = physical_group(name, row.File)
                entry = dict(file=f"{name}.h5", dataset=name, dataset_id=int(row.Dataset_id),
                             metadata_id=int(row.Id), recording_id=int(row.Id), original_recording=row.File,
                             raw_exists=raw_path.is_file(), h5_exists=data is not None, h5_shape=str(shape),
                             h5_dtype=str(data.dtype) if data is not None else "missing", label=row.Label,
                             duplicate_original_path=int(row.Id) in duplicate_ids,
                             shared_resolved_raw_path=int(row.Id) in shared_raw_ids,
                             shared_raw_inode=int(row.Id) in shared_inode_ids,
                             group=group, metadata_rate=row.Sample_rate, metadata_channel=row.Channel,
                             metadata_length=row.Sample_lenth, observed_length=shape[0] if shape else 0,
                             observed_channels=shape[1] if len(shape) >= 2 else 0)
                local_records.append(entry)
                if not raw_path.is_file() or data is None:
                    anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="raw or H5 record missing", observation=str(raw_path)))
                    continue
                if not shape or any(size == 0 for size in shape):
                    anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="zero-sized H5 record", observation=str(shape)))
                if data.dtype.kind not in "fiu":
                    anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="nonnumeric H5 record", observation=str(data.dtype)))
                if len(shape) >= 2 and (shape[1] != row.Channel or shape[0] != row.Sample_lenth):
                    anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="metadata shape mismatch", observation=f"metadata=({row.Sample_lenth},{row.Channel}); H5={shape}"))
                # One whole-record comparison per actual channel/rate/modality
                # category. This is explicit sampled provenance, not an assertion
                # of bytewise equivalence of the full 83+ GiB collection.
                if name == "RM_007_MFPT":
                    original = loadmat(raw_path, squeeze_me=True, struct_as_record=False)["bearing"]
                    raw_rate = float(original.sr)
                    raw_signal = np.asarray(original.gs).reshape(-1)
                    exact = np.array_equal(raw_signal, data[:].reshape(-1))
                    acquisition.append(dict(file=f"{name}.h5", metadata_id=int(row.Id), observation="embedded MAT sr and entire gs compared to cache",
                                            raw_rate_hz=raw_rate, metadata_rate_hz=row.Sample_rate,
                                            cache_equals_raw=exact, details="no resampling where exact=True"))
                    if raw_rate != row.Sample_rate:
                        anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="raw versus metadata sample-rate mismatch", observation=f"raw sr={raw_rate}; metadata={row.Sample_rate}; cache_equals_raw={exact}"))
                category = (int(row.Dataset_id), shape[1:] if shape else (), row.Sample_rate,
                            "temperature" if Path(row.File).name.startswith("temp_") else "signal")
                if category in probe_categories:
                    continue
                probe_categories.add(category)
                if name == "RM_027_PU":
                    original = loadmat(raw_path, squeeze_me=True, struct_as_record=False)[raw_path.stem]
                    signal = np.atleast_1d(original.Y)[6]
                    axis = np.atleast_1d(original.X)[int(signal.XIndex) - 1]
                    timestamps = np.asarray(axis.Data, dtype=float).reshape(-1)
                    acquisition.append(dict(file=f"{name}.h5", metadata_id=int(row.Id), observation="original vibration_1 time axis and units; native reader discards time axis",
                                            raw_rate_hz=(timestamps.size - 1) / (timestamps[-1] - timestamps[0]), metadata_rate_hz=row.Sample_rate,
                                            details=compact(dict(unit=np.asarray(signal.Unit).tolist(), samples=int(np.asarray(signal.Data).size),
                                                                 timestamps=int(timestamps.size), duration=float(timestamps[-1] - timestamps[0]),
                                                                 dt_quantiles=np.quantile(np.diff(timestamps), [0, .5, 1]).tolist()))))
                probe: dict[str, object] = dict(file=f"{name}.h5", dataset_id=int(row.Dataset_id), metadata_id=int(row.Id),
                                               original_recording=row.File, h5_shape=str(shape), reader=f"src/data_factory/reader/{name}.py")
                try:
                    raw = np.asarray(reader(str(raw_path)))
                    cached = data[:]
                    probe.update(raw_shape=str(raw.shape), raw_dtype=str(raw.dtype))
                    if raw.dtype.kind not in "fiu" or cached.dtype.kind not in "fiu":
                        raise TypeError(f"nonnumeric signal: raw dtype={raw.dtype}, H5 dtype={cached.dtype}")
                    if cached.ndim == 3 and cached.shape[-1] == 1:
                        cached = cached[..., 0]
                    equal = raw.shape == cached.shape and np.array_equal(raw, cached, equal_nan=True)
                    probe.update(raw_shape=str(raw.shape), values_equal=equal, finite=bool(np.isfinite(raw).all()),
                                 status="PASS" if equal else "MISMATCH", error="")
                    if not equal:
                        anomalies.append(dict(file=f"{name}.h5", metadata_id=key, issue="native raw reader differs from H5", observation=f"reader={raw.shape}; H5={cached.shape}"))
                except (OSError, ValueError, TypeError, KeyError, IndexError, ImportError) as exc:
                    probe.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", values_equal=False)
                probes.append(probe)

            for dataset_id, part in subset.groupby("Dataset_id", sort=True):
                these = [r for r in local_records if r["dataset_id"] == dataset_id]
                failures = [x.strip() for x in note["blockers"].split(";")]
                if missing:
                    failures.append("metadata IDs missing from H5")
                if extra:
                    failures.append("unmapped H5 IDs retained in anomaly table; container population not fully reconciled")
                if part.Label.isna().any() or (part.Label < 0).any():
                    failures.append("missing/negative task labels cannot be silently dropped")
                if any(not r["raw_exists"] for r in these):
                    failures.append("one or more original recording files missing")
                if any(r["duplicate_original_path"] or r["shared_resolved_raw_path"] or r["shared_raw_inode"] for r in these):
                    failures.append("duplicate original Name/File or shared resolved raw path; Id alone does not establish distinct recordings")
                if any(r["observed_channels"] != r["metadata_channel"] or r["observed_length"] != r["metadata_length"] for r in these):
                    failures.append("metadata versus H5 length/channel mismatch")
                if any(r["observed_channels"] == 0 for r in these):
                    failures.append("zero-channel H5 records")
                labels = sorted({str(r["label"]) for r in these})
                group_counts = {label: len({r["group"] for r in these if str(r["label"]) == label})
                                if all(r["group"] is not None for r in these if str(r["label"]) == label) else "unknown"
                                for label in labels}
                episode = fixed_episode(these)
                selected_probes = [p for p in probes if p["file"] == f"{name}.h5" and p["dataset_id"] == dataset_id]
                if any(p["status"] != "PASS" for p in selected_probes):
                    failures.append("sampled native raw/cache equivalence failed; inspect RAW_CACHE_COMPARISON.csv")
                rates = compact(sorted(float(x) for x in part.Sample_rate.dropna().unique()))
                effective = rates
                effective_evidence = "metadata and documentary acquisition rate; sampled unchanged native raw/cache records, not full collection verification"
                if name == "RM_007_MFPT":
                    effective = compact(sorted({a["raw_rate_hz"] for a in acquisition if a["file"] == f"{name}.h5"}))
                    effective_evidence = "all23 embedded raw sr fields; all23 full gs arrays equal H5, so cache did not resample"
                elif name == "RM_027_PU":
                    effective = "unknown as a uniform grid: nominal64000Hz; sampled Id47567 average64205.445Hz with nonuniform time increments"
                    effective_evidence = "RAW_ACQUISITION_CHECKS.csv; H5 retains raw Y values but drops X time axis"
                elif name in {"RM_001_CWRU", "RM_006_THU"}:
                    effective = "unresolved: raw native-rate/channel transformation conflicts; metadata rates are claims only"
                elif name == "RM_018_THU24":
                    effective = "approximately3125Hz exported filtered channels; native pre-filter rate unknown"
                    effective_evidence = "CSV sample interval319.99999192us and Filtered CH1/CH2 headers"
                qualifications.append(dict(
                    file=f"{name}.h5", dataset=name, dataset_id=int(dataset_id), metadata_id=compact([int(x) for x in part.Id]),
                    metadata_id_count=len(part), h5_key_count=len(keys), missing_metadata_ids=compact(missing), extra_h5_ids=compact(extra),
                    container_semantics="one container with bearing dataset_id=9 and gear dataset_id=15; same corpus" if name == "RM_010_SEU" else "one named corpus container; per-Id original acquisition records, not a single statistical observation",
                    raw_record_recoverable=all(r["raw_exists"] and r["h5_exists"] for r in these),
                    raw_files_present=sum(bool(r["raw_exists"]) for r in these),
                    duplicate_original_path_count=sum(bool(r["duplicate_original_path"]) for r in these),
                    shared_resolved_raw_path_count=sum(bool(r["shared_resolved_raw_path"]) for r in these),
                    shared_raw_inode_count=sum(bool(r["shared_raw_inode"]) for r in these),
                    recording_mapping="metadata.Id -> str(Id) H5 key -> metadata.Name/File; RECORD_INVENTORY.csv",
                    window_mapping="no experimental windows generated; future channel/start/end intervals must retain metadata.Id and documented group",
                    group_key=note["group"], num_groups_per_class=compact(group_counts),
                    record_count_per_class=compact(Counter(str(r["label"]) for r in these)),
                    native_rate=note["native"], metadata_sample_rates=rates, effective_rate=effective,
                    effective_rate_evidence=effective_evidence,
                    physical_unit=note["unit"], sensor_position=note["sensor"],
                    channel=compact(sorted({int(r["observed_channels"]) for r in these})),
                    preprocessing=note["processing"], cache_provenance=f"root attributes={dict(h5.attrs)}; {attrs_count}/{len(keys)} records have attributes; sampled native-reader equivalence in RAW_CACHE_COMPARISON.csv",
                    common_support="not_constructed: complete source-acquisition qualification unavailable",
                    increment_support="not_constructed: available/unavailable atoms not established",
                    support_evidence=note["support"], license=note["license"], evidence=note["evidence"],
                    eligible=not failures, exclusion_reason="; ".join(failures), **episode,
                ))
        records.extend(local_records)
        print(f"{name}: {len(local_records)} metadata records inspected", flush=True)

    dataset_ids = [int(q["dataset_id"]) for q in qualifications]
    folds = []
    for q in qualifications:
        candidate_sources = [int(s["dataset_id"]) for s in qualifications if s["dataset"] != q["dataset"]]
        qualified_sources = [int(s["dataset_id"]) for s in qualifications if s["eligible"] and s["dataset"] != q["dataset"]]
        reasons = [str(q["exclusion_reason"])] if not q["eligible"] else []
        if len(qualified_sources) < 2:
            reasons.append("fewer than two qualified independent source corpora")
        reasons.append("source-defined nonempty common/increment sets and target binary support not established")
        if q["episode_status"] != "PASS":
            reasons.append(f"episode {q['episode_status']}: {q['episode_reason']}")
        folds.append(dict(target=q["dataset"], dataset_id=q["dataset_id"], fold=f"LOCO_dataset_{q['dataset_id']}",
                          candidate_source_dataset_ids=compact(candidate_sources), qualified_source_dataset_ids=compact(qualified_sources),
                          same_corpus_dataset_ids_excluded=compact([int(s["dataset_id"]) for s in qualifications if s["dataset"] == q["dataset"] and s["dataset_id"] != q["dataset_id"]]),
                          common_support=q["common_support"], increment_support=q["increment_support"], target_common="unknown",
                          target_increment="unknown", source_availability="unresolved; no Nyquist-only inference", target_device_evidence=q["support_evidence"],
                          episode_seed=1729, support_budget="5 original recordings per class", support_record_count=q["support_record_count"],
                          support_group_count=q["support_group_count"], query_group_count=q["query_group_count"],
                          episode_status=q["episode_status"], eligible=False, status="failure" if q["episode_status"] == "FAIL" else "ineligible",
                          exclusion_reason="; ".join(reasons)))
    pd.DataFrame(qualifications).to_csv(output / "DATASET_QUALIFICATION.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "FOLD_ELIGIBILITY.csv", index=False)
    pd.DataFrame(records).to_csv(output / "RECORD_INVENTORY.csv", index=False)
    pd.DataFrame(probes).to_csv(output / "RAW_CACHE_COMPARISON.csv", index=False)
    pd.DataFrame(anomalies).to_csv(output / "DATA_ANOMALIES.csv", index=False)
    pd.DataFrame(acquisition).to_csv(output / "RAW_ACQUISITION_CHECKS.csv", index=False)
    elapsed = time.monotonic() - started
    print(compact(dict(files=len(FILES), datasets=len(dataset_ids), records=len(records), probes=len(probes),
                       qualified_datasets=sum(bool(q["eligible"]) for q in qualifications), eligible_folds=0,
                       elapsed_seconds=round(elapsed, 3), output=str(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/user/data/PHMbenchdata/PHM-Vibench"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tii_local_j1"))
    args = parser.parse_args()
    run(args.data_root, args.output_dir)
