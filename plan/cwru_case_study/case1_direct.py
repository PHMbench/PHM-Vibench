#!/usr/bin/env python3
"""
Case 1: Direct Few-Shot Learning (No Pretraining)

This script implements the baseline case where the model is trained directly
on few-shot episodes without any pretraining. This serves as the baseline
for comparing the effectiveness of pretraining strategies.

Tasks:
1. Fault Diagnosis (4-class classification)
2. Anomaly Detection (binary classification)
3. Signal Prediction (next-window forecasting)

Author: PHM-Vibench Development Team
Date: September 2025
"""

import os
import sys
import time
import torch
import numpy as np
from datetime import datetime

# Import common utilities
from common_utils import (
    # Constants
    N_CHANNELS, N_CLASSES_DIAG, N_CLASSES_ANOM, N_SUPPORT, N_QUERY,
    TASKS_TO_RUN, FINETUNE_EPOCHS, LEARNING_RATE, device,

    # Functions
    setup_logger, log_system_info, init_common_setup,
    load_cwru_data, create_few_shot_episodes,
    train_classification, train_prediction, evaluate_classification_metrics,
    save_results,

    # Models
    DirectFewShotModel
)

def main():
    """Main execution function for Case 1"""
    start_time = time.time()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/case1_direct_{timestamp}.log"
    logger = setup_logger("Case1_Direct", log_file)

    logger.info("="*80)
    logger.info("CASE 1: DIRECT FEW-SHOT LEARNING (NO PRETRAINING)")
    logger.info("="*80)

    # Log system information
    log_system_info(logger)

    # Initialize setup
    device = init_common_setup()
    logger.info(f"Using device: {device}")

    # Results storage
    results = {
        'case': 'Case 1 - Direct Few-Shot Learning',
        'timestamp': timestamp,
        'tasks': {},
        'hyperparameters': {
            'n_support': N_SUPPORT,
            'n_query': N_QUERY,
            'learning_rate': LEARNING_RATE,
            'epochs': FINETUNE_EPOCHS,
            'n_classes_diag': N_CLASSES_DIAG,
            'n_classes_anom': N_CLASSES_ANOM
        }
    }

    try:
        # ==================== DATA LOADING ====================
        logger.info("\n" + "-"*60)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("-"*60)

        signals, diag_labels, anom_labels, file_ids, scaler = load_cwru_data(logger)
        logger.info(f"Loaded {len(signals)} signal windows")
        logger.info(f"Signal shape: {signals.shape}")

        # ==================== MODEL INITIALIZATION ====================
        logger.info("\n" + "-"*60)
        logger.info("STEP 2: MODEL INITIALIZATION")
        logger.info("-"*60)

        model = DirectFewShotModel(N_CHANNELS, TASKS_TO_RUN, N_CLASSES_DIAG, N_CLASSES_ANOM).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model created: DirectFewShotModel")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log model architecture
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        logger.info(f"Backbone parameters: {backbone_params:,}")

        for task_name, head in model.heads.items():
            head_params = sum(p.numel() for p in head.parameters())
            logger.info(f"Head[{task_name}] parameters: {head_params:,}")

        # ==================== TASK 1: FAULT DIAGNOSIS ====================
        if TASKS_TO_RUN.get('diagnosis', False):
            logger.info("\n" + "-"*60)
            logger.info("STEP 3: FAULT DIAGNOSIS TASK")
            logger.info("-"*60)

            # Create few-shot episodes for diagnosis
            support_x, support_y, query_x, query_y = create_few_shot_episodes(
                signals, diag_labels, N_SUPPORT, N_QUERY, N_CLASSES_DIAG, logger
            )

            # Convert to tensors
            support_x = torch.FloatTensor(support_x)
            support_y = torch.LongTensor(support_y)
            query_x = torch.FloatTensor(query_x)
            query_y = torch.LongTensor(query_y)

            logger.info(f"Diagnosis episode: Support {support_x.shape}, Query {query_x.shape}")

            # Train diagnosis task
            diag_losses, diag_accuracies = train_classification(
                model, support_x, support_y, query_x, query_y,
                'diagnosis', epochs=FINETUNE_EPOCHS, lr=LEARNING_RATE, logger=logger
            )

            # Evaluate final metrics
            model.eval()
            with torch.no_grad():
                query_logits = model(query_x.to(device), 'diagnosis')
                query_preds = torch.argmax(query_logits, dim=1).cpu().numpy()

            diag_metrics = evaluate_classification_metrics(query_y.numpy(), query_preds, logger)

            # Store results
            results['tasks']['diagnosis'] = {
                'final_accuracy': diag_accuracies[-1],
                'final_loss': diag_losses[-1],
                'training_losses': diag_losses,
                'training_accuracies': diag_accuracies,
                'metrics': diag_metrics,
                'support_shape': support_x.shape,
                'query_shape': query_x.shape
            }

            logger.info(f"✅ Diagnosis task completed - Final accuracy: {diag_accuracies[-1]:.4f}")

        # ==================== TASK 2: ANOMALY DETECTION ====================
        if TASKS_TO_RUN.get('anomaly', False):
            logger.info("\n" + "-"*60)
            logger.info("STEP 4: ANOMALY DETECTION TASK")
            logger.info("-"*60)

            # Create few-shot episodes for anomaly detection
            support_x, support_y, query_x, query_y = create_few_shot_episodes(
                signals, anom_labels, N_SUPPORT, N_QUERY, N_CLASSES_ANOM, logger
            )

            # Convert to tensors
            support_x = torch.FloatTensor(support_x)
            support_y = torch.LongTensor(support_y)
            query_x = torch.FloatTensor(query_x)
            query_y = torch.LongTensor(query_y)

            logger.info(f"Anomaly episode: Support {support_x.shape}, Query {query_x.shape}")

            # Train anomaly detection task
            anom_losses, anom_accuracies = train_classification(
                model, support_x, support_y, query_x, query_y,
                'anomaly', epochs=FINETUNE_EPOCHS, lr=LEARNING_RATE, logger=logger
            )

            # Evaluate final metrics
            model.eval()
            with torch.no_grad():
                query_logits = model(query_x.to(device), 'anomaly')
                query_preds = torch.argmax(query_logits, dim=1).cpu().numpy()

            anom_metrics = evaluate_classification_metrics(query_y.numpy(), query_preds, logger)

            # Store results
            results['tasks']['anomaly'] = {
                'final_accuracy': anom_accuracies[-1],
                'final_loss': anom_losses[-1],
                'training_losses': anom_losses,
                'training_accuracies': anom_accuracies,
                'metrics': anom_metrics,
                'support_shape': support_x.shape,
                'query_shape': query_x.shape
            }

            logger.info(f"✅ Anomaly detection completed - Final accuracy: {anom_accuracies[-1]:.4f}")

        # ==================== TASK 3: SIGNAL PREDICTION ====================
        if TASKS_TO_RUN.get('prediction', False):
            logger.info("\n" + "-"*60)
            logger.info("STEP 5: SIGNAL PREDICTION TASK")
            logger.info("-"*60)

            # Create prediction episodes (current window -> next window)
            logger.info("Creating prediction episodes...")

            # Filter normal signals for prediction task (anomaly label = 0)
            normal_mask = anom_labels == 0
            normal_signals = signals[normal_mask]

            if len(normal_signals) < N_SUPPORT + N_QUERY:
                logger.warning(f"Not enough normal signals for prediction task: {len(normal_signals)}")
                logger.warning("Skipping prediction task")
            else:
                # Create current->next window pairs
                indices = np.random.permutation(len(normal_signals) - 1)  # -1 because we need next window
                total_needed = N_SUPPORT + N_QUERY

                if len(indices) < total_needed:
                    total_needed = len(indices)
                    logger.warning(f"Reduced samples to {total_needed} due to data constraints")

                selected_indices = indices[:total_needed]

                current_windows = normal_signals[selected_indices]
                next_windows = normal_signals[selected_indices + 1]

                # Split into support and query
                support_current = current_windows[:N_SUPPORT]
                support_next = next_windows[:N_SUPPORT]
                query_current = current_windows[N_SUPPORT:total_needed]
                query_next = next_windows[N_SUPPORT:total_needed]

                # Convert to tensors
                support_x = torch.FloatTensor(support_current)
                support_y = torch.FloatTensor(support_next)
                query_x = torch.FloatTensor(query_current)
                query_y = torch.FloatTensor(query_next)

                logger.info(f"Prediction episode: Support {support_x.shape}->{support_y.shape}")
                logger.info(f"                   Query {query_x.shape}->{query_y.shape}")

                # Train prediction task
                pred_losses, pred_mse_values = train_prediction(
                    model, support_x, support_y, query_x, query_y,
                    epochs=FINETUNE_EPOCHS, lr=LEARNING_RATE, logger=logger
                )

                # Store results
                results['tasks']['prediction'] = {
                    'final_mse': pred_mse_values[-1],
                    'final_loss': pred_losses[-1],
                    'training_losses': pred_losses,
                    'training_mse': pred_mse_values,
                    'support_shape': support_x.shape,
                    'query_shape': query_x.shape
                }

                logger.info(f"✅ Prediction task completed - Final MSE: {pred_mse_values[-1]:.6f}")

        # ==================== RESULTS SUMMARY ====================
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time

        logger.info("\n" + "="*80)
        logger.info("CASE 1 RESULTS SUMMARY")
        logger.info("="*80)

        if 'diagnosis' in results['tasks']:
            diag_acc = results['tasks']['diagnosis']['final_accuracy']
            logger.info(f"🔧 Fault Diagnosis:     {diag_acc:.4f} accuracy")

        if 'anomaly' in results['tasks']:
            anom_acc = results['tasks']['anomaly']['final_accuracy']
            logger.info(f"🚨 Anomaly Detection:   {anom_acc:.4f} accuracy")

        if 'prediction' in results['tasks']:
            pred_mse = results['tasks']['prediction']['final_mse']
            logger.info(f"📈 Signal Prediction:   {pred_mse:.6f} MSE")

        logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")

        # Save results
        results_file = f"results/case1_results_{timestamp}.pkl"
        save_results(results, results_file, logger)

        logger.info(f"\n✅ Case 1 completed successfully!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📝 Log saved to: {log_file}")

        return results

    except Exception as e:
        logger.error(f"❌ Case 1 failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise e

if __name__ == "__main__":
    results = main()

    # Print final summary to console
    print("\n" + "="*80)
    print("CASE 1 EXECUTION COMPLETED")
    print("="*80)

    for task_name, task_results in results['tasks'].items():
        if task_name in ['diagnosis', 'anomaly']:
            acc = task_results['final_accuracy']
            print(f"{task_name.title():20s}: {acc:.4f} accuracy")
        elif task_name == 'prediction':
            mse = task_results['final_mse']
            print(f"{task_name.title():20s}: {mse:.6f} MSE")

    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print("Check logs/ and results/ directories for detailed output")