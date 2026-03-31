# PHM-Vibench Verified Dataset Report

This document provides a **verified, ground-truth** reference for all datasets in the PHM-Vibench framework.
Information is synthesized from three sources:
1.  **Metadata (`metadata.xlsx`)**: Primary source for Fs, Operating Conditions, and System IDs.
2.  **H5 Data Inspection**: Ground truth for Channel Counts and Sample Lengths (`inspect_h5_structure.py`).
3.  **Reader Code (`src/data_factory/reader/*.py`)**: Verification of data loading logic.

> [!IMPORTANT]
> **RM_009_Paderborn** is missing from both `metadata.xlsx` and the `src/data_factory/reader` directory.
> **RM_025_KAIST** and **RM_026_HUST23** are present as placeholders only (`# TODO` in code).

## 1. Master Dataset Summary

| Dataset | System ID | Fs (Hz) | Channels (Verified) | Domains | ID Range | Description/Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RM_001_CWRU** | 1 | 12000, 48000 | **2** | 4 | 1 - 155 | Classic bearing faults. Metadata lists both 12k/48k. |
| **RM_002_XJTU** | 2 | 25600 | **2** | 3 | 164 - 9379 | Run-to-failure. 35Hz-40Hz operating conditions. |
| **RM_003_FEMTO** | 3 | 25600 | **2** | 3 | 9380 - 36525 | PRONOSTIA platform. Load 1, 2, 3. |
| **RM_004_IMS** | 4 | 20000 | **8** | 1 | 36526 - 45988 | Run-to-failure. 4 bearings x 2 channels (accumulated). |
| **RM_005_Ottawa23** | 5 | *Time Varying* | **3** | 28 | 45990 - 46050 | Time-varying speed/load. Metadata Fs is empty/null. |
| **RM_006_THU** | 6 | 20480, 40960 | **1** | 1 | 46231 - 46261 | Single channel vibration. |
| **RM_007_MFPT** | 7 | 48828, 97656 | **1** | 10 | 46282 - 46304 | Variable loads (0-300 lbs). High sampling rate. |
| **RM_008_UNSW** | 8 | 51200 | **6** | 6 | 46305 - 46932 | 6 channels (AccH, AccV, Enc, etc). Frequency domain variations. |
| **RM_009_Paderborn** | N/A | N/A | N/A | N/A | N/A | **MISSING FROM CODEBASE & METADATA** |
| **RM_010_SEU** | 9, 15 | 5120 | **8** | 2 | 46933 - 46952 | Gearbox dataset. 20Hz/30Hz. |
| **RM_011_pump** | N/A | *File Dependent* | N/A | N/A | N/A | **Metadata Missing**. Python reader exists. |
| **RM_015_susu** | 11 | 31175 | **3** | 2 | 46985 - 46994 | 18Hz, 20Hz operations. |
| **RM_016_JNU** | 12 | 50000 | **1** | 3 | 46995 - 47006 | 600-1000 rpm. |
| **RM_017_Ottawa19** | 13 | 200000 | **2** | 4 | 47007 - 47042 | Very high Fs (200k). Speed ramping. |
| **RM_018_THU24** | 14 | 3125 | **2** | 1 | 47044 - 47058 | Low Fs (3.125k). |
| **RM_019_HIT** | N/A | N/A | N/A | N/A | N/A | **Metadata Missing**. Reader reads `.npy` files directly. |
| **RM_020_DIRG** | 16 | 51200, 102400 | **6** | 17 | 47064 - 47243 | High speed (100-400Hz). |
| **RM_023_HIT23** | 17 | 51200 | **1** | 3 | 47246 - 47287 | 3 simple speed conditions. |
| **RM_024_JUST** | 18 | 50000 | **7** | 9 | 47288 - 47566 | 7 channels. Low speed (2-12 rpm). |
| **RM_025_KAIST** | - | - | - | - | - | **PLACEHOLDER ONLY** (`# TODO` in reader). |
| **RM_026_HUST23** | - | - | - | - | - | **PLACEHOLDER ONLY** (`# TODO` in reader). |
| **RM_027_PU** | 20 | 64000 | **3** | 4 | 47567 - 50126 | Paderborn University? (Maybe implies RM_009 successor?) |
| **RM_031_HUST24** | 19 | 25600 | **3** | 11 | 47408 - 47506 | 20Hz - 80Hz sweep. |

---

## 2. Detailed Domain Mappings (One-by-One Verification)

This section maps every `Domain_id` found in `metadata.xlsx` to its verified Description and Condition.

### RM_001_CWRU (System ID: 1)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 1797 rpm | 0HP load |
| **1.0** | 1772 rpm | 1HP load |
| **2.0** | 1750 rpm | 2HP load |
| **3.0** | 1730 rpm | 3HP load |

### RM_002_XJTU (System ID: 2)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 35Hz | 12kN Load |
| **1.0** | 37.5Hz | 11kN Load |
| **2.0** | 40Hz | 10kN Load |

### RM_003_FEMTO (System ID: 3)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | Load 1 | 1800 rpm, 4000 N |
| **1.0** | Load 2 | 1650 rpm, 4200 N |
| **2.0** | Load 3 | 1500 rpm, 5000 N |

### RM_004_IMS (System ID: 4)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 2000 rpm | 6000 lbs |

### RM_005_Ottawa23 (System ID: 5)
*Note: Highly complex time-varying dataset. Domains 0-27.*
| Domain Range | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0 - 27.0** | (Various) | Combinations of Speed (Incre/Decre/Const) and Load (Incre/Decre/Const). Reference metadata for specific ID. |

### RM_006_THU (System ID: 6)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | - | Single operating condition in metadata. |

### RM_007_MFPT (System ID: 7)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0 - 2.0** | 270 lbs | Baseline (Healthy) conditions |
| **3.0 - 9.0** | (Various) | Fault conidtions under loads: 0, 50, 100, 150, 200, 250, 300 lbs |

### RM_008_UNSW (System ID: 8)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 6 Hz | - |
| **1.0** | 10 Hz | - |
| **2.0** | 15 Hz | - |
| **3.0** | 20 Hz | - |
| **4.0** | 25 Hz | - |
| **5.0** | 30 Hz | - |

### RM_010_SEU (System ID: 9, 15)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 20Hz | Load 0V |
| **1.0** | 30Hz | Load 2V |

### RM_015_susu (System ID: 11)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 18 Hz | - |
| **1.0** | 20 Hz | - |

### RM_016_JNU (System ID: 12)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 600 rpm | - |
| **1.0** | 800 rpm | - |
| **2.0** | 1000 rpm | - |

### RM_017_Ottawa19 (System ID: 13)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | Speed Decreasing | - |
| **1.0** | Speed Increasing | - |
| **2.0** | Speed Incr/Decr | - |
| **3.0** | Steady Speed | - |

### RM_018_THU24 (System ID: 14)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | - | - |

### RM_020_DIRG (System ID: 16)
*High Speed Test Rig*
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0 - 16.0** | 100Hz - 400Hz | Variable speeds and Loads (0Nm, 4Nm...) |

### RM_023_HIT23 (System ID: 17)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 600 rpm | - |
| **1.0** | 900 rpm | - |
| **2.0** | 1200 rpm | - |

### RM_024_JUST (System ID: 18)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | - | - |
| **1.0 - 3.0** | 2 rpm | Loads: 0N, 30N, 60N |
| **4.0 - 6.0** | 6 rpm | Loads: 0N, 30N, 60N |
| **7.0 - 9.0** | 12 rpm | Loads: 0N, 30N, 60N |

### RM_027_PU (System ID: 20)
*Paderborn University Data (Likely separate from RM_009)*
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0** | 1500rpm | 0.1Nm, 1000N |
| **1.0** | 900rpm | 0.7Nm, 1000N |
| **2.0** | 1500rpm | 0.7Nm, 1000N |
| **3.0** | 1500rpm | 0.7Nm, 400N |

### RM_031_HUST24 (System ID: 19)
| Domain ID | Description | Condition Notes |
| :--- | :--- | :--- |
| **0.0 - 9.0** | 20Hz - 80Hz | 5Hz increments |
| **10.0** | 0-40-0 Hz | Variable frequency |

---

## 3. Reference: H5 File Inspection Counts
*Confirmed Channel Counts (Ch) from `inspect_h5_structure.py` analysis.*

*   **RM_001_CWRU.h5**: 2 Ch
*   **RM_002_XJTU.h5**: 2 Ch
*   **RM_003_FEMTO.h5**: 2 Ch
*   **RM_004_IMS.h5**: 8 Ch
*   **RM_005_Ottawa23.h5**: 3 Ch
*   **RM_006_THU.h5**: 1 Ch
*   **RM_007_MFPT.h5**: 1 Ch
*   **RM_008_UNSW.h5**: 6 Ch
*   **RM_010_SEU.h5**: 8 Ch
*   **RM_015_susu.h5**: 3 Ch
*   **RM_016_JNU.h5**: 1 Ch
*   **RM_017_Ottawa19.h5**: 2 Ch
*   **RM_018_THU24.h5**: 2 Ch
*   **RM_020_DIRG.h5**: 6 Ch
*   **RM_023_HIT23.h5**: 1 Ch
*   **RM_024_JUST.h5**: 7 Ch
*   **RM_027_PU.h5**: 3 Ch
*   **RM_031_HUST24.h5**: 3 Ch
