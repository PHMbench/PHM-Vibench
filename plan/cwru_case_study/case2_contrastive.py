#!/usr/bin/env python3
"""
Case 2: Contrastive Pretraining + Few-Shot Learning

This script implements contrastive pretraining followed by few-shot fine-tuning.
The encoder is pretrained using SimCLR-style contrastive learning, then fine-tuned
on downstream tasks with the encoder unfrozen for adaptation.

Pipeline:
1. Phase 1: Contrastive pretraining on unlabeled data
2. Phase 2: Few-shot fine-tuning on labeled episodes

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
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Import common utilities
from common_utils import (
    # Constants
    N_CHANNELS, N_CLASSES_DIAG, N_CLASSES_ANOM, N_SUPPORT, N_QUERY,
    TASKS_TO_RUN, PRETRAIN_EPOCHS, FINETUNE_EPOCHS, LEARNING_RATE, BATCH_SIZE, device,

    # Functions
    setup_logger, log_system_info, init_common_setup,
    load_cwru_data, create_few_shot_episodes,
    train_classification, train_prediction, evaluate_classification_metrics,
    save_results, contrastive_loss,

    # Models
    DirectFewShotModel
)

def create_augmented_pairs(signals, logger):
    """Create augmented pairs for contrastive learning"""
    if logger:
        logger.info("Creating augmented pairs for contrastive learning...")

    batch_size = len(signals)

    # Original signals
    signals_1 = signals.copy()

    # Create augmented versions
    signals_2 = signals.copy()

    # Apply augmentations to signals_2
    for i in range(batch_size):
        signal = signals_2[i]

        # Augmentation 1: Gaussian noise (σ=0.1)
        noise = np.random.normal(0, 0.1, signal.shape)
        signal += noise

        # Augmentation 2: Amplitude scaling (0.8-1.2x)
        scale = np.random.uniform(0.8, 1.2)
        signal *= scale

        # Augmentation 3: Time shifting (±50 samples)
        shift = np.random.randint(-50, 51)
        if shift > 0:
            signal = np.concatenate([signal[shift:], signal[:shift]], axis=0)
        elif shift < 0:
            signal = np.concatenate([signal[shift:], signal[:shift]], axis=0)

        signals_2[i] = signal

    if logger:
        logger.info(f"Created {batch_size} augmented pairs")
    return signals_1, signals_2

def pretrain_contrastive(model, signals, epochs, lr, logger):
    """Pretrain encoder using contrastive learning"""
    logger.info(f"Starting contrastive pretraining for {epochs} epochs...")

    # Setup optimizer for encoder only
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(signals))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    losses = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (batch_signals,) in enumerate(dataloader):
            batch_signals = batch_signals.numpy()

            # Create augmented pairs
            signals_1, signals_2 = create_augmented_pairs(batch_signals, logger if batch_idx == 0 else None)

            # Convert to tensors and move to device
            signals_1 = torch.FloatTensor(signals_1).to(device)
            signals_2 = torch.FloatTensor(signals_2).to(device)

            # Forward pass with projection heads
            embeddings_1, _ = model.backbone(signals_1, use_projection=True)
            embeddings_2, _ = model.backbone(signals_2, use_projection=True)

            # Compute contrastive loss
            loss = contrastive_loss(embeddings_1, embeddings_2, temperature=0.1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(loss.item())

        # Update learning rate
        scheduler.step()

        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'checkpoints/best_contrastive_model.pth')
        else:
            patience_counter += 1

        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch:3d}/{epochs}: Loss {avg_loss:.4f}, LR {current_lr:.2e}, Patience {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
            break

    # Load best model
    if os.path.exists('checkpoints/best_contrastive_model.pth'):
        model.load_state_dict(torch.load('checkpoints/best_contrastive_model.pth'))
        logger.info("Loaded best contrastive model")

    logger.info(f"Contrastive pretraining completed. Final loss: {losses[-1]:.4f}")
    return losses

def main():
    """Main execution function for Case 2"""
    start_time = time.time()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/case2_contrastive_{timestamp}.log"
    logger = setup_logger("Case2_Contrastive", log_file)

    logger.info("="*80)
    logger.info("CASE 2: CONTRASTIVE PRETRAINING + FEW-SHOT LEARNING")
    logger.info("="*80)

    # Log system information
    log_system_info(logger)

    # Initialize setup
    device = init_common_setup()
    logger.info(f"Using device: {device}")

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Results storage
    results = {
        'case': 'Case 2 - Contrastive Pretraining + Few-Shot',
        'timestamp': timestamp,
        'pretraining': {},
        'tasks': {},
        'hyperparameters': {
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'finetune_epochs': FINETUNE_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'n_support': N_SUPPORT,
            'n_query': N_QUERY,
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
        backbone_params = sum(p.numel() for p in model.backbone.parameters())

        logger.info(f"Model created: DirectFewShotModel with UnifiedEncoder")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Backbone parameters: {backbone_params:,}")

        for task_name, head in model.heads.items():
            head_params = sum(p.numel() for p in head.parameters())
            logger.info(f"Head[{task_name}] parameters: {head_params:,}")

        # ==================== PHASE 1: CONTRASTIVE PRETRAINING ====================
        logger.info("\n" + "-"*60)
        logger.info("STEP 3: CONTRASTIVE PRETRAINING PHASE")
        logger.info("-"*60)

        # Use all available signals for pretraining (unlabeled)
        pretrain_losses = pretrain_contrastive(
            model, signals, PRETRAIN_EPOCHS, LEARNING_RATE, logger
        )

        # Store pretraining results
        results['pretraining'] = {
            'losses': pretrain_losses,
            'final_loss': pretrain_losses[-1],
            'epochs_trained': len(pretrain_losses),
            'num_samples': len(signals)
        }

        logger.info(f"✅ Contrastive pretraining completed")
        logger.info(f"   Final loss: {pretrain_losses[-1]:.4f}")
        logger.info(f"   Epochs trained: {len(pretrain_losses)}")

        # ==================== PHASE 2: FEW-SHOT FINE-TUNING ====================
        logger.info("\n" + "-"*60)
        logger.info("STEP 4: FEW-SHOT FINE-TUNING PHASE")
        logger.info("-"*60)

        # ==================== TASK 1: FAULT DIAGNOSIS ====================
        if TASKS_TO_RUN.get('diagnosis', False):
            logger.info("\n" + "."*40)
            logger.info("TASK 1: FAULT DIAGNOSIS")
            logger.info("."*40)

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
            logger.info("Fine-tuning with UNFROZEN encoder (adaptive learning)")

            # Setup adaptive learning rates (encoder: 0.1x, heads: 1.0x)
            encoder_params = list(model.backbone.parameters())
            head_params = list(model.heads['diagnosis'].parameters())

            optimizer = torch.optim.Adam([
                {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained encoder
                {'params': head_params, 'lr': LEARNING_RATE}            # Full LR for new head
            ])

            criterion = torch.nn.CrossEntropyLoss()

            # Custom training loop with adaptive learning
            losses, accuracies = [], []

            for epoch in range(FINETUNE_EPOCHS):
                model.train()
                optimizer.zero_grad()

                # Train on support set
                support_logits = model(support_x.to(device), 'diagnosis')
                loss = criterion(support_logits, support_y.to(device))
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Evaluate on query set
                model.eval()
                with torch.no_grad():
                    query_logits = model(query_x.to(device), 'diagnosis')
                    query_preds = torch.argmax(query_logits, dim=1)
                    accuracy = (query_preds == query_y.to(device)).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch:3d}: Loss {loss.item():.4f}, Accuracy {accuracy:.4f}")

            # Final evaluation
            model.eval()
            with torch.no_grad():
                query_logits = model(query_x.to(device), 'diagnosis')
                query_preds = torch.argmax(query_logits, dim=1).cpu().numpy()

            diag_metrics = evaluate_classification_metrics(query_y.numpy(), query_preds, logger)

            # Store results
            results['tasks']['diagnosis'] = {
                'final_accuracy': accuracies[-1],
                'final_loss': losses[-1],
                'training_losses': losses,
                'training_accuracies': accuracies,
                'metrics': diag_metrics,
                'support_shape': support_x.shape,
                'query_shape': query_x.shape,
                'fine_tuning_method': 'unfrozen_adaptive_lr'
            }

            logger.info(f"✅ Diagnosis task completed - Final accuracy: {accuracies[-1]:.4f}")

        # ==================== TASK 2: ANOMALY DETECTION ====================
        if TASKS_TO_RUN.get('anomaly', False):
            logger.info("\n" + "."*40)
            logger.info("TASK 2: ANOMALY DETECTION")
            logger.info("."*40)

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

            # Setup adaptive learning rates
            encoder_params = list(model.backbone.parameters())
            head_params = list(model.heads['anomaly'].parameters())

            optimizer = torch.optim.Adam([
                {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},
                {'params': head_params, 'lr': LEARNING_RATE}
            ])

            criterion = torch.nn.CrossEntropyLoss()

            # Custom training loop
            losses, accuracies = [], []

            for epoch in range(FINETUNE_EPOCHS):
                model.train()
                optimizer.zero_grad()

                support_logits = model(support_x.to(device), 'anomaly')
                loss = criterion(support_logits, support_y.to(device))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    query_logits = model(query_x.to(device), 'anomaly')
                    query_preds = torch.argmax(query_logits, dim=1)
                    accuracy = (query_preds == query_y.to(device)).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch:3d}: Loss {loss.item():.4f}, Accuracy {accuracy:.4f}")

            # Final evaluation
            model.eval()
            with torch.no_grad():
                query_logits = model(query_x.to(device), 'anomaly')
                query_preds = torch.argmax(query_logits, dim=1).cpu().numpy()

            anom_metrics = evaluate_classification_metrics(query_y.numpy(), query_preds, logger)

            # Store results
            results['tasks']['anomaly'] = {
                'final_accuracy': accuracies[-1],
                'final_loss': losses[-1],
                'training_losses': losses,
                'training_accuracies': accuracies,
                'metrics': anom_metrics,
                'support_shape': support_x.shape,
                'query_shape': query_x.shape,
                'fine_tuning_method': 'unfrozen_adaptive_lr'
            }

            logger.info(f"✅ Anomaly detection completed - Final accuracy: {accuracies[-1]:.4f}")

        # ==================== TASK 3: SIGNAL PREDICTION ====================
        if TASKS_TO_RUN.get('prediction', False):
            logger.info("\n" + "."*40)
            logger.info("TASK 3: SIGNAL PREDICTION")
            logger.info("."*40)

            # Create prediction episodes
            normal_mask = anom_labels == 0
            normal_signals = signals[normal_mask]

            if len(normal_signals) < N_SUPPORT + N_QUERY:
                logger.warning(f"Not enough normal signals for prediction: {len(normal_signals)}")
                logger.warning("Skipping prediction task")
            else:
                # Create current->next window pairs
                indices = np.random.permutation(len(normal_signals) - 1)
                total_needed = N_SUPPORT + N_QUERY

                if len(indices) < total_needed:
                    total_needed = len(indices)

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

                # Setup adaptive learning rates
                encoder_params = list(model.backbone.parameters())
                head_params = list(model.heads['prediction'].parameters())

                optimizer = torch.optim.Adam([
                    {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},
                    {'params': head_params, 'lr': LEARNING_RATE}
                ])

                criterion = torch.nn.MSELoss()

                # Custom training loop
                losses, mse_values = [], []

                for epoch in range(FINETUNE_EPOCHS):
                    model.train()
                    optimizer.zero_grad()

                    support_pred = model(support_x.to(device), 'prediction')
                    loss = criterion(support_pred, support_y.to(device))
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        query_pred = model(query_x.to(device), 'prediction')
                        mse = F.mse_loss(query_pred, query_y.to(device)).item()

                    losses.append(loss.item())
                    mse_values.append(mse)

                    if epoch % 10 == 0:
                        logger.info(f"Epoch {epoch:3d}: Loss {loss.item():.6f}, Query MSE {mse:.6f}")

                # Store results
                results['tasks']['prediction'] = {
                    'final_mse': mse_values[-1],
                    'final_loss': losses[-1],
                    'training_losses': losses,
                    'training_mse': mse_values,
                    'support_shape': support_x.shape,
                    'query_shape': query_x.shape,
                    'fine_tuning_method': 'unfrozen_adaptive_lr'
                }

                logger.info(f"✅ Prediction task completed - Final MSE: {mse_values[-1]:.6f}")

        # ==================== RESULTS SUMMARY ====================
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time

        logger.info("\n" + "="*80)
        logger.info("CASE 2 RESULTS SUMMARY")
        logger.info("="*80)

        # Pretraining results
        logger.info(f"🔄 Contrastive Pretraining: {results['pretraining']['final_loss']:.4f} final loss")
        logger.info(f"   Epochs: {results['pretraining']['epochs_trained']}")
        logger.info(f"   Samples: {results['pretraining']['num_samples']:,}")

        # Task results
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
        results_file = f"results/case2_results_{timestamp}.pkl"
        save_results(results, results_file, logger)

        logger.info(f"\n✅ Case 2 completed successfully!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📝 Log saved to: {log_file}")

        return results

    except Exception as e:
        logger.error(f"❌ Case 2 failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise e

if __name__ == "__main__":
    results = main()

    # Print final summary to console
    print("\n" + "="*80)
    print("CASE 2 EXECUTION COMPLETED")
    print("="*80)

    print(f"Pretraining loss: {results['pretraining']['final_loss']:.4f}")

    for task_name, task_results in results['tasks'].items():
        if task_name in ['diagnosis', 'anomaly']:
            acc = task_results['final_accuracy']
            print(f"{task_name.title():20s}: {acc:.4f} accuracy")
        elif task_name == 'prediction':
            mse = task_results['final_mse']
            print(f"{task_name.title():20s}: {mse:.6f} MSE")

    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print("Check logs/ and results/ directories for detailed output")