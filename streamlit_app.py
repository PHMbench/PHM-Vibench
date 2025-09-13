import argparse
import importlib
import os
import tempfile
from typing import Any, Dict, List, Optional


from src.utils import (
    build_env_phmbench,
    build_env_traditional,
    save_config,
    load_config,
)

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    st = None


def list_config_files() -> List[str]:
    """Recursively list yaml config files under the configs directory."""
    cfg_files: List[str] = []
    for root, _, files in os.walk("configs"):
        for name in files:
            if name.endswith(".yaml"):
                cfg_files.append(os.path.join(root, name))
    return sorted(cfg_files)


def list_pipeline_modules() -> List[str]:
    """Return available pipeline modules under src directory."""
    pipelines: List[str] = []
    for name in os.listdir("src"):
        if name.startswith("Pipeline") and name.endswith(".py"):
            pipelines.append(name[:-3])
    return sorted(pipelines)


def run_pipeline(
    config: Dict[str, Any],
    pipeline_name: str = "Pipeline_01_default",
    fs_config_path: Optional[str] = None,
) -> Any:
    """Run a pipeline using a temporary config file."""
    module = importlib.import_module(f"src.{pipeline_name}")
    import yaml  # local import
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f)
        temp_path = f.name
    args = argparse.Namespace(
        config_path=temp_path,
        fs_config_path=fs_config_path,
        notes="",
    )
    results = module.pipeline(args)
    os.remove(temp_path)
    return results


def parse_metadata_file(uploaded_file: Any):
    """Parse uploaded metadata file into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        import pandas as pd  # local import
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception:  # pragma: no cover - UI error
        if st:
            st.sidebar.error("读取元数据失败")
        return None


def choose_ids_ui(ids: List[str]) -> List[List[str]]:
    """Let the user select train/val/test ids from list."""
    if st is None:
        return [[], [], []]
    train = st.sidebar.multiselect("选择 train ids", ids)
    val = st.sidebar.multiselect("选择 val ids", ids)
    test = st.sidebar.multiselect("选择 test ids", ids)
    return [train, val, test]


def build_traditional_ui() -> Optional[Dict[str, Any]]:
    """Upload train/val/test files via sidebar."""
    if st is None:
        return None
    train_files = st.sidebar.file_uploader(
        "上传训练文件", accept_multiple_files=True
    )
    val_files = st.sidebar.file_uploader(
        "上传验证文件", accept_multiple_files=True
    )
    test_files = st.sidebar.file_uploader(
        "上传测试文件", accept_multiple_files=True
    )
    if train_files and val_files and test_files:
        return build_env_traditional(train_files, val_files, test_files)
    return None


def build_phmbench_ui() -> Optional[Dict[str, Any]]:
    """Upload metadata and choose ids via sidebar."""
    if st is None:
        return None
    metadata_file = st.sidebar.file_uploader(
        "上传 metadata.csv/xlsx", type=["csv", "xlsx"]
    )
    data_dir = st.sidebar.text_input("h5 文件目录")
    meta_df = parse_metadata_file(metadata_file)
    if meta_df is not None:
        ids = meta_df.iloc[:, 0].astype(str).tolist()
        train_ids, val_ids, test_ids = choose_ids_ui(ids)
        if train_ids and val_ids and test_ids:
            return build_env_phmbench(
                meta_df, data_dir, train_ids, val_ids, test_ids
            )
    return None


def cast_value(original: Any, new_val: Any) -> Any:
    """Cast a UI value back to the original type."""
    if isinstance(original, bool):
        return bool(new_val)
    if isinstance(original, int) and not isinstance(original, bool):
        return int(new_val)
    if isinstance(original, float):
        return float(new_val)
    if isinstance(original, (list, dict)):
        try:
            import yaml  # local import
            return yaml.safe_load(new_val)
        except Exception:
            return original
    return type(original)(new_val)


def config_to_yaml(cfg: Dict[str, Any]) -> str:
    """Dump configuration dictionary to a YAML string."""
    import yaml

    return yaml.safe_dump(cfg, allow_unicode=True)


def download_config(cfg: Dict[str, Any]) -> None:
    """Render a download button for the provided configuration."""
    if st is None:
        return
    yaml_str = config_to_yaml(cfg)
    st.download_button(
        "💾 下载配置 | Download Config",
        yaml_str,
        file_name="edited_config.yaml",
    )


def render_field(label: str, value: Any, key: str) -> Any:
    """Render a single config field based on value type."""
    if st is None:
        return value
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if isinstance(value, int) and not isinstance(value, bool):
        return st.number_input(label, value=value, step=1, key=key)
    if isinstance(value, float):
        return st.number_input(label, value=value, format="%f", key=key)
    if isinstance(value, (list, dict)):
        import yaml  # local import
        return st.text_input(label, value=yaml.safe_dump(value), key=key)
    return st.text_input(label, value=str(value), key=key)


def edit_section(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create UI inputs for a config section and return updated values."""
    if st is None:
        return cfg
    updated: Dict[str, Any] = {}
    with st.expander(f"{name} Parameters"):
        for k, v in cfg.items():
            widget_key = f"{name}_{k}"
            new_v = render_field(k, v, key=widget_key)
            updated[k] = cast_value(v, new_v)
    return updated


def select_environment() -> Optional[Dict[str, Any]]:
    """Sidebar widgets for uploading data and building environment dict."""
    if st is None:
        return None
    st.sidebar.header("数据加载方式")
    load_mode = st.sidebar.radio(
        "选择模式",
        ["传统文件上传模式", "PHMbench元数据模式"],
        index=0,
    )

    if load_mode == "传统文件上传模式":
        return build_traditional_ui()
    return build_phmbench_ui()
    return None


def select_pipeline() -> (str, Optional[str]):
    """Sidebar widgets to choose pipeline and optional FS config."""
    if st is None:
        return "Pipeline_01_default", None
    st.sidebar.header("实验流水线")
    pipelines = list_pipeline_modules()
    selected = st.sidebar.selectbox("选择 pipeline", pipelines)
    fs_path = None
    if selected == "Pipeline_02_pretrain_fewshot":
        fs_path = st.sidebar.selectbox("Few-shot 配置", list_config_files())
    return selected, fs_path


def load_base_config() -> Dict[str, Any]:
    """Load a base YAML config selected from sidebar."""
    if st is None:
        return load_config(list_config_files()[0])
    st.sidebar.header("实验配置")
    config_files = list_config_files()
    selected_cfg_path = st.sidebar.selectbox("选择配置文件", config_files)
    return load_config(selected_cfg_path)


def update_config(
    base_cfg: Dict[str, Any],
    sections: Dict[str, Dict[str, Any]],
    env: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge edited sections and environment into base config."""
    cfg = dict(base_cfg)
    for sec, val in sections.items():
        cfg[sec] = val
    if env is not None:
        cfg.setdefault("environment", {}).update(env)
    return cfg


def display_results(result: Any) -> None:
    """Render experiment results in tabs."""
    if st is None:
        return
    import pandas as pd
    tab1, _, _, tab4 = st.tabs(["指标总结", "训练曲线", "预测详情", "原始日志"])
    with tab1:
        if isinstance(result, list) and result:
            df = pd.DataFrame(result)
            st.dataframe(df)
        else:
            st.write("无结果")
    with tab4:
        st.code(str(result))


def run_app() -> None:
    if st is None:
        raise RuntimeError("streamlit is required to run the app")
    st.set_page_config(page_title="PHMbench UI", layout="wide")
    st.title("PHMbench 实验管理器")

    uploaded_env = select_environment()
    pipeline, fs_path = select_pipeline()
    base_cfg = load_base_config()

    updated_sections = {}
    for sec in ["data", "model", "task", "trainer"]:
        if sec in base_cfg:
            updated_sections[sec] = edit_section(sec, base_cfg[sec])

    col1, col2, col3 = st.columns(3)
    save_clicked = col1.button("💾 保存配置 | Save Config")
    refresh_clicked = col2.button("🔄 刷新 | Refresh")
    run_clicked = col3.button("🚀 开始实验 | Run Experiment")

    if save_clicked or run_clicked:
        final_cfg = update_config(base_cfg, updated_sections, uploaded_env)

    if save_clicked:
        download_config(final_cfg)

    if refresh_clicked:
        st.rerun()

    if run_clicked:
        with st.spinner("实验进行中..."):
            result = run_pipeline(final_cfg, pipeline, fs_path)
        st.success("实验完成")
        display_results(result)


if __name__ == "__main__":
    run_app()
