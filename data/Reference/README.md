# Reference papers: geometry, sensors and source Reader caveats

This index describes **facts found in original source publications or named external
publisher pages** and constraints of the fixed 2026-10-26 legacy Reader. It does not
change or replace any third-party paper in this directory; it is not an approval to
redistribute their contents. A source-paper parameter is *not* automatically the
value measured in every converted record. Actual raw file/metadata and physical-asset
binding are separate acceptance checks.

The 21 folders below match the present [PHMskills Profile inventory](https://github.com/liq22/phm-agent-benchmark/tree/dev/.agent/skills/phmskills/profiles).
The fields `rolling_elements, ball_diameter_m, pitch_diameter_m, contact_angle_rad`
are only runnable after **all four** values and the bearing mechanism have been
verified for the particular acquired asset. Never infer a diagnosis, bearing geometry
or sensor type from a fault-coded `File`, `Name`, `Label` or target group.

| Paper/source folder | Corroborated physical or sensing fact | Still requires validation |
|---|---|---|
| [CWRU](RM_001_CWRU/) | Official CWRU [DE SKF 6205](https://engineering.case.edu/bearingdatacenter/bearing-information) ball 0.3126 in, pitch 1.537 in; FE SKF 6203 ball 0.2656 in, pitch 1.122 in. Official fault orders are distinct. | DE/FE bearing identity, exact contact angle and original 12/48-kHz variant; reader exports DE then optional FE, not base |
| [XJTU-SY](RM_002_XJTU/) | Original [publisher](https://biaowang.tech/xjtu-sy-bearing-datasets/) and [independent Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC9321675/) identify LDK UER204: 8 balls, 7.92-mm diameter, 34.55-mm pitch, 0°. Two PCB 352C33 accelerometers: CSV 0 horizontal, 1 vertical. | Match physical tested bearing and whether numeric CSV samples are raw volts or calibrated acceleration; actual sample rate from file |
| [PRONOSTIA/FEMTO](RM_003_FEMTO/) | Two orthogonal accelerometers; independent [NSK 6804RS table](https://www.mdpi.com/1996-1073/14/8/2322) indicates 13 balls, d=3.5 mm and mean race diameter 25.6 mm. | Contact angle, pitch-circle definition and reader's three exported source fields (CSV 4–6) |
| [IMS](RM_004_IMS/) | NASA IMS, Rexnord ZA-2115 **double-row roller** bearing; 16 rolling elements per row, approximately 8.4-mm element and 71.5-mm pitch, 15.17° in [independent comparison](https://www.mdpi.com/2076-3417/13/10/5977). | Per-trial 8/4 channel bearing assignments, 20 kHz vs derived 20.48 kHz ambiguity; ball-bearing formulas not supported |
| [Ottawa23](RM_005_Ottawa23/) | [Original data article](https://www.sciencedirect.com/science/article/pii/S2352340923004456) distinguishes acceleration, sound, speed, load, temperature; the Reader retains only raw CSV columns 0,1,4. | Verify physical column order, sensor units, source version and bearing contact angle/variant before binding |
| [THU22](RM_006_THU/) | Piezoelectric harvesting/sensing acquisition, not a guaranteed calibrated acceleration in m/s². | Original voltage-to-acceleration transfer, measurement axis and bearing geometry |
| [MFPT](RM_007_MFPT/) | Independent [NICE bearing report](https://www.mdpi.com/2076-3417/15/3/1257) gives 8 elements, ball 0.235 in, pitch 1.245 in and nominal angle 0°. | Bundled PDF is an envelope tutorial; exact MFPT specimen, units and distinct original sample-rate source must be checked |
| [UNSW](RM_008_UNSW/) | [Original dataset release v3](https://data.mendeley.com/datasets/h4df4mgrfb/3): accH/accV (stored in V), enc1/enc2 (1024 pulses/rev), loadCell (stored in V), tacho (one pulse/rev). | Reader omits absent fields, shifting positional output indices; use actual MAT variables and version |
| [SEU](RM_010_SEU/) | Separate bearing/gearbox experiments; eight exported columns include heterogeneous motor and gearbox measurement roles. | Determine subexperiment and channel calibration; no one universal bearing geometry |
| [NU205 pump](RM_011_pump/) | Cylindrical-roller source and mixed instrumentation; Excel Reader loads last worksheet then drops first row. | Reader's NumPy `.float()` bug, actual worksheet units, no ordinary ball-bearing method |
| [JNU](RM_016_JNU/) | N205/NU205 roller-bearing experiments; published [sensor PCB MA352A60](https://pmc.ncbi.nlm.nih.gov/articles/PMC12564828/) used vertically at 50 kHz. | Source-specific roller count/pitch and physical mechanism; not identical to a ball-bearing configuration |
| [Ottawa19](RM_017_Ottawa19/) | [Mendeley v1](https://data.mendeley.com/datasets/v43hmbwxpm/1) has 36 recordings and Channel_1 acceleration, Channel_2 encoder at 200 kHz; [v2](https://data.mendeley.com/datasets/v43hmbwxpm/2) has 60 recordings, encoder 1024 CPR. | Variable-speed trajectory representation and source version; cannot treat encoder pulses as constant shaft speed |
| [THU24](RM_018_THU24/) | Hybrid triboelectric–piezoelectric voltage channels, Reader selects raw columns 4 and 10. | Source units, gain, actual output directions; not interchangeable with THU22 |
| [HIT inter-shaft](RM_019_HIT/) | High-/low-pressure rotor system with displacement and casing-acceleration instrumentation. | Relative shaft kinematics, actual NPY array axes and bearing subtype |
| [DIRG](RM_020_DIRG/) | Original [MSSP study](https://www.sciencedirect.com/science/article/pii/S0888327018306800) describes two triaxial accelerometers. | Per-series bearing mechanism, A1/A2 sensor positions and MAT variable axes |
| [HIT23](RM_023_HIT23/) | Separate source from inter-shaft HIT. | Reader concatenates MAT arrays without a frozen variable-to-channel order |
| [JUST slewing](RM_024_JUST/) | [Original Mendeley](https://data.mendeley.com/datasets/hwg8v5j8t6/1) declares **7 measurement columns**: four vertical, two horizontal vibration, one acoustic emission. | Reader currently drops the first column; inspect original CSV before preserving/dropping a possible timestamp; cross-roller geometry unsupported |
| [KAIST](RM_025_KAIST/) | [Original source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036499/): 4 accelerometers, 1 microphone, 2 thermocouples, 3 current transformers across distinct acquisition settings. | Current Reader `# TODO`; exported channel clocks/units and per-asset bearing mechanism unknown |
| [HUST23](RM_026_HUST23/) | [Original 2023 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10327369/) gives KG 6204–6208 rolling-element count and ball diameter, PCB 325C33 vertical acceleration at 51.2 kHz. | **Pitch and contact angle are not specified**; Reader `# TODO`; 0/200/400 W is motor power, not radial force |
| [PU](RM_027_PU/) | Original Paderborn Reference files; two independently reviewed geometry candidates are already present in PHMskills. | Asset-to-geometry assignment, channel current/current/vibration and gain must be bound for actual specimen |
| [HUST24](RM_031_HUST24/) | [Original author README](https://github.com/CHAOZHAO-1/HUSTbearing-dataset/blob/main/README.md) gives ER-16K 9 balls, 7.94-mm ball, **38.52 mm explicitly labeled shaft diameter**, 10 fixed-speed + one variable-speed condition. | **Do not reuse shaft diameter as pitch**; independently published ER-16K dimensions conflict; source Reader outputs three tab-separated fields after header removal |

## Reader-specific acceptance

For a particular input, inspect `raw/[Name]/[File]` under a legal local
dataset root and record **actual original field → reader expression → output
index → physical modality/axis/unit and calibration**. A printed channel count
or paper test-bench drawing does not prove current converted H5 channel identity.

Known source-level risks: `RM_011_pump` calls nonexistent `ndarray.float()`;
`RM_024_JUST` drops the first column despite the seven-column source claim;
`RM_008_UNSW` compresses missing MAT variables; `RM_005_Ottawa23` selects
mixed modalities by column position; `RM_023_HIT23` concatenates MAT arrays
without a fixed name-to-output mapping; KAIST/HUST23 Reader files contain TODOs.

Keep these as **explicitly unresolved** until a representative original input
confirms the intended behavior. No private fault labels, sensor calibration,
bearing-geometry or dataset-population claim can be supplied by paper titles.
