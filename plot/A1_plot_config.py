import matplotlib.pyplot as plt
import seaborn as sns
# Assuming scienceplots is correctly installed
# If not, you might need to install it using pip
import scienceplots
from matplotlib import font_manager
import os

# colors = ['#4D4398','#984696', '#747474','#42453B' , '#74C6BE', 'm']  # 莫兰迪配色


def set_chinese_font(font_path='/home/user/.fonts/simhei.ttf'):
    """
    设置 matplotlib 使用中文字体。

    参数:
    - font_path (str): 字体文件的路径。默认为当前工作目录下的 'simhei.ttf'。

    使用方法:
    >>> set_chinese_font('path/to/simhei.ttf')
    """
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        print(f"字体文件未找到: {font_path}")
        return
    
    # 添加字体到字体管理器
    font_manager.fontManager.addfont(font_path)
    
    # 获取字体名称
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    
    # 全局设置字体
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    print(f"已设置字体为: {font_name}")


def configure_matplotlib(style='ieee', font_lang='en', seaborn_theme=False, font_scale=1.4):
    """
    Configure matplotlib and seaborn (optional) settings for plots.

    Parameters:
    - style: str, matplotlib style to use. Default is 'ieee'.
    - font_lang: str, language for font settings. 'en' for English (Times New Roman), 'cn' for Chinese (SimHei).
    - seaborn_theme: bool, whether to apply seaborn theme settings. Default is False.
    - font_scale: float, scaling factor for fonts if seaborn theme is applied.
    """
    # Define font settings
    fonts = {
        'en': {'family': 'Times New Roman', 'weight': 'normal', 'size': 12},
        'cn': {'family': 'simhei', 'weight': 'normal', 'size': 12},
    }

    # Apply matplotlib style
    plt.style.use(['science', style])

    # Configure fonts based on language
    if font_lang == 'cn':
        set_chinese_font()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese characters correctly
    plt.rcParams['font.family'] = fonts[font_lang]['family']
    plt.rcParams['font.size'] = fonts[font_lang]['size']
    plt.rcParams['font.weight'] = fonts[font_lang]['weight']

    # Optionally apply seaborn theme
    if seaborn_theme:
        sns.set_theme(style="white", font='sans-serif', font_scale=font_scale)

if __name__ == '__main__':
    # Example usage
    configure_matplotlib(style='ieee', font_lang='en', seaborn_theme=False, font_scale=1.4)

