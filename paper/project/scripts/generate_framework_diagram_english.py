#!/usr/bin/env python3
"""
English version of Neural-Symbolic XFD Framework Diagram
International version without Chinese font issues
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import argparse
import os

def create_framework_diagram_english(output_path='fig_neuralsymbolic_overview_english.png', dpi=300):
    """
    Create English version of Neural-Symbolic XFD Framework Diagram
    """

    plt.figure(figsize=(16, 12), dpi=dpi)
    plt.style.use('default')

    # Color scheme
    layer_colors = {
        'signal': '#E3F2FD',
        'feature': '#BBDEFB',
        'symbolic': '#90CAF9',
        'linguistic': '#64B5F6',
        'input': '#FFF3E0',
        'connection': '#757575'
    }

    # Define layers
    layers = [
        {'name': 'Linguistic Layer', 'y': 0.85, 'color': layer_colors['linguistic'], 'height': 0.1},
        {'name': 'Symbolic Layer', 'y': 0.65, 'color': layer_colors['symbolic'], 'height': 0.1},
        {'name': 'Feature Layer', 'y': 0.45, 'color': layer_colors['feature'], 'height': 0.1},
        {'name': 'Signal Layer', 'y': 0.25, 'color': layer_colors['signal'], 'height': 0.1},
        {'name': 'Raw Signals Input', 'y': 0.05, 'color': layer_colors['input'], 'height': 0.1}
    ]

    # Subproject modules
    subprojects = {
        '1D-2D_Fusion': {
            'layer': 'signal',
            'x': 0.15,
            'label': '1D-2D\nFusion',
            'color': '#4CAF50'
        },
        'MOE': {
            'layer': 'signal',
            'x': 0.35,
            'label': 'MOE\nExperts',
            'color': '#FF9800'
        },
        'Operator_Attention': {
            'layer': 'signal',
            'x': 0.55,
            'label': 'Operator\nAttention',
            'color': '#9C27B0'
        },
        'Fuzzy': {
            'layer': 'feature',
            'x': 0.75,
            'label': 'Fuzzy\nProcessing',
            'color': '#F44336'
        },
        'Cross_modal': {
            'layer': 'feature',
            'x': 0.15,
            'label': 'Cross-modal\nAlignment',
            'color': '#2196F3'
        },
        'Expert_Features': {
            'layer': 'feature',
            'x': 0.35,
            'label': 'Expert\nFeatures',
            'color': '#FF5722'
        },
        'Attention_Weights': {
            'layer': 'feature',
            'x': 0.55,
            'label': 'Attention\nWeights',
            'color': '#673AB7'
        },
        'Statistical': {
            'layer': 'feature',
            'x': 0.75,
            'label': 'Statistical\nFeatures',
            'color': '#E91E63'
        },
        'Fuzzy_Rules': {
            'layer': 'symbolic',
            'x': 0.2,
            'label': 'Fuzzy\nRules',
            'color': '#795548'
        },
        'Expert_Logic': {
            'layer': 'symbolic',
            'x': 0.4,
            'label': 'Expert\nLogic',
            'color': '#607D8B'
        },
        'Knowledge_Graph': {
            'layer': 'symbolic',
            'x': 0.6,
            'label': 'Knowledge\nGraph',
            'color': '#3F51B5'
        },
        'Evaluation': {
            'layer': 'symbolic',
            'x': 0.8,
            'label': 'Evaluation\nProtocols',
            'color': '#009688'
        },
        'LLM_Explainer': {
            'layer': 'linguistic',
            'x': 0.25,
            'label': 'LLM\nExplainer',
            'color': '#CDDC39'
        },
        'Expert_Explainer': {
            'layer': 'linguistic',
            'x': 0.5,
            'label': 'Expert\nExplainer',
            'color': '#FFC107'
        },
        'Unified_Interface': {
            'layer': 'linguistic',
            'x': 0.75,
            'label': 'Unified\nInterface',
            'color': '#8BC34A'
        }
    }

    # Draw layer backgrounds
    for layer in layers:
        if layer['name'] == 'Raw Signals Input':
            rect = FancyBboxPatch(
                (0.05, layer['y']), 0.9, layer['height'],
                boxstyle="round,pad=0.01",
                facecolor=layer['color'],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7
            )
        else:
            rect = FancyBboxPatch(
                (0.05, layer['y']), 0.9, layer['height'],
                boxstyle="round,pad=0.01",
                facecolor=layer['color'],
                edgecolor='black',
                linewidth=1.2,
                alpha=0.6
            )
        plt.gca().add_patch(rect)

        # Add layer labels
        plt.text(0.02, layer['y'] + layer['height']/2, layer['name'],
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                transform=plt.gca().transData)

    # Draw subproject modules
    for name, config in subprojects.items():
        # Find corresponding layer
        layer_y = None
        layer_mapping = {
            'signal': 'Signal Layer',
            'feature': 'Feature Layer',
            'symbolic': 'Symbolic Layer',
            'linguistic': 'Linguistic Layer'
        }

        target_layer_name = layer_mapping.get(config['layer'])
        for layer in layers:
            if layer['name'] == target_layer_name:
                layer_y = layer['y'] + layer['height']/2
                break

        if layer_y is not None:
            rect = FancyBboxPatch(
                (config['x'] - 0.06, layer_y - 0.03), 0.12, 0.06,
                boxstyle="round,pad=0.005",
                facecolor=config['color'],
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            plt.gca().add_patch(rect)

            plt.text(config['x'], layer_y, config['label'],
                    fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    color='white' if config['color'] in ['#FF5722', '#F44336', '#9C27B0', '#673AB7', '#3F51B5', '#795548'] else 'black')

    # Draw data flow arrows (bottom-up)
    arrow_y_positions = [0.15, 0.35, 0.55, 0.75]
    for y_pos in arrow_y_positions:
        arrow = ConnectionPatch(
            (0.5, y_pos), (0.5, y_pos + 0.1),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc=layer_colors['connection'],
            linewidth=2, alpha=0.7
        )
        plt.gca().add_patch(arrow)

    # Add data flow labels
    plt.text(0.52, 0.2, 'Signal Processing', fontsize=10, style='italic')
    plt.text(0.52, 0.4, 'Feature Extraction', fontsize=10, style='italic')
    plt.text(0.52, 0.6, 'Symbolic Reasoning', fontsize=10, style='italic')
    plt.text(0.52, 0.8, 'Linguistic Explanation', fontsize=10, style='italic')

    # Add constraint arrows (top-down, dashed)
    constraint_arrow_x = 0.95
    for i in range(len(layers) - 2, 0, -1):
        arrow = ConnectionPatch(
            (constraint_arrow_x, layers[i]['y'] + layers[i]['height']),
            (constraint_arrow_x, layers[i-1]['y']),
            "data", "data",
            arrowstyle="->", shrinkA=3, shrinkB=3,
            mutation_scale=15, fc='red',
            linewidth=1.5, alpha=0.6,
            linestyle='--'
        )
        plt.gca().add_patch(arrow)

    # Add constraint label
    plt.text(0.92, 0.5, 'Theoretical\nConstraints', fontsize=9, color='red',
            ha='center', va='center', rotation=90)

    # Set axis
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    # Add title
    plt.title('Neural-Symbolic Explainable Fault Diagnosis Unified Framework',
             fontsize=18, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        patches.Patch(color='#4CAF50', label='1D-2D Fusion'),
        patches.Patch(color='#FF9800', label='MOE Experts'),
        patches.Patch(color='#9C27B0', label='Operator Attention'),
        patches.Patch(color='#F44336', label='Fuzzy System'),
        patches.Patch(color='#2196F3', label='Cross-modal Alignment'),
        patches.Patch(color='#3F51B5', label='Knowledge Graph'),
        patches.Patch(color='#CDDC39', label='LLM Explanation'),
        patches.Patch(color='#757575', label='Data Flow')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              ncol=2, fontsize=8, framealpha=0.9)

    # Add explanation text
    explanation_text = """
    Data Flow: Signal → Feature → Symbolic → Linguistic (solid arrows)
    Constraint Flow: Upper layers constrain lower layers for consistency (dashed arrows)
    """
    plt.text(0.98, 0.02, explanation_text, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Save image
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"English framework diagram saved to: {output_path}")

    # Also save PDF version
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate English Neural-Symbolic XFD Framework Diagram')
    parser.add_argument('--output', '-o',
                       default='fig_neuralsymbolic_overview_english.png',
                       help='Output image path')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Image resolution (default: 300)')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate diagram
    create_framework_diagram_english(args.output, args.dpi)

if __name__ == '__main__':
    main()