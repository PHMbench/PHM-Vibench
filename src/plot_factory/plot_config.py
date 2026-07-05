"""Shared matplotlib / seaborn styling for plot_factory.

Migrated from ``plot/A1_plot_config.py``. No external project dependencies —
this module can be imported standalone (it only needs matplotlib + seaborn +
scienceplots, which are plotting-stack requirements).
"""

import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

try:
    import scienceplots  # noqa: F401  (registers the 'science' style on import)
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


def set_chinese_font(font_path: str = "/home/user/.fonts/simhei.ttf") -> None:
    """Configure matplotlib to render Chinese characters.

    Parameters
    ----------
    - font_path (str): Path to a TTF font file. Defaults to ``~/.fonts/simhei.ttf``.
    """
    if not os.path.exists(font_path):
        print(f"字体文件未找到: {font_path}")
        return

    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False  # render minus signs correctly
    print(f"已设置字体为: {font_name}")


def configure_matplotlib(
    style: str = "ieee",
    font_lang: str = "en",
    seaborn_theme: bool = False,
    font_scale: float = 1.4,
) -> None:
    """Configure matplotlib (and optionally seaborn) for publication plots.

    Parameters
    ----------
    - style: matplotlib style to apply on top of ``science``. Default ``'ieee'``.
    - font_lang: ``'en'`` (Liberation Sans) or ``'cn'`` (SimHei).
    - seaborn_theme: if True, apply a seaborn ``white`` theme at ``font_scale``.
    - font_scale: font scaling factor for the seaborn theme.
    """
    fonts = {
        "en": {"family": "Liberation Sans", "weight": "normal", "size": 12},
        "cn": {"family": "simhei", "weight": "normal", "size": 12},
    }

    plt.style.use(["science", style] if _HAS_SCIENCEPLOTS else ["seaborn-v0_8-whitegrid"])

    if font_lang == "cn":
        set_chinese_font()
        plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.family"] = fonts[font_lang]["family"]
    plt.rcParams["font.size"] = fonts[font_lang]["size"]
    plt.rcParams["font.weight"] = fonts[font_lang]["weight"]

    if seaborn_theme:
        sns.set_theme(style="white", font="sans-serif", font_scale=font_scale)

    if not _HAS_SCIENCEPLOTS:
        warnings.warn(
            "scienceplots not installed; falling back to a default matplotlib "
            "style. Install with `pip install scienceplots` for publication styling.",
            stacklevel=2,
        )


if __name__ == "__main__":
    configure_matplotlib(style="ieee", font_lang="en", seaborn_theme=False, font_scale=1.4)
