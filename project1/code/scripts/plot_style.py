import matplotlib.pyplot as plt

def setup_economist_style():
    """
    Sets up a plotting style inspired by 'The Economist'.
    """
    # Color palette
    ECONOMIST_RED = '#E3120B'
    ECONOMIST_BLUE = '#006BA2'
    ECONOMIST_DARK_GREY = '#333333'
    ECONOMIST_LIGHT_GREY = '#CBCBCB'
    
    plt.rcParams.update({
        # Figure and background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        
        # Spines
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.0,
        'axes.edgecolor': ECONOMIST_DARK_GREY,
        
        # Grid
        'axes.grid': False,
        
        # Fonts and sizes
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'xtick.major.size': 5,
        'xtick.major.width': 1.0,
        'ytick.labelsize': 10,
        'ytick.major.size': 5,
        'ytick.major.width': 1.0,
        'legend.fontsize': 10,
        'legend.frameon': False,
        
        # Line widths
        'lines.linewidth': 2.0,
        
        # Colors
        'axes.prop_cycle': plt.cycler(color=[ECONOMIST_BLUE, ECONOMIST_RED, '#3EBCD2', '#37A635', '#FB832D', '#822433', '#EBBDE5']),
        'text.color': ECONOMIST_DARK_GREY,
        'axes.labelcolor': ECONOMIST_DARK_GREY,
        'xtick.color': ECONOMIST_DARK_GREY,
        'ytick.color': ECONOMIST_DARK_GREY,
    })

def add_economist_signature(ax, title, subtitle=None):
    """
    Adds a stylized title and optional subtitle in The Economist style.
    """
    ax.set_title(title, loc='left', pad=40, fontsize=16, fontweight='bold', color='black')
    if subtitle:
        ax.text(0, 1.08, subtitle, transform=ax.transAxes, fontsize=12, color='#555555')
    
    # Add the signature red rectangle top-left
    ax.annotate('', xy=(0, 1.18), xytext=(0.04, 1.18), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color='#E3120B', linewidth=5))
