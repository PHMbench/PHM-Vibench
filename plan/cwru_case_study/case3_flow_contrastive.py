#!/usr/bin/env python3
"""
Case 3: Flow + Contrastive Joint Training + Few-Shot Learning

This script implements joint flow matching and contrastive pretraining followed by
few-shot fine-tuning. The encoder learns both generative (flow) and discriminative
(contrastive) representations simultaneously through a combined loss function.

Pipeline:
1. Phase 1: Joint flow + contrastive pretraining (flow_loss + 0.3 * contrastive_loss)
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
    WINDOW_SIZE, FLOW_AVAILABLE,

    # Functions
    setup_logger, log_system_info, init_common_setup,
    load_cwru_data, create_few_shot_episodes,
    train_classification, train_prediction, evaluate_classification_metrics,
    save_results, contrastive_loss, SimpleFlowModel,

    # Models
    DirectFewShotModel
)

# Import FlowLoss if available
if FLOW_AVAILABLE:
    try:
        from src.task_factory.Components.flow import FlowLoss
        print("✅ FlowLoss imported successfully")
    except ImportError:
        FLOW_AVAILABLE = False
        print("⚠️ FlowLoss import failed, using simplified version")

def create_augmented_pairs(signals, logger):
    """Create augmented pairs for contrastive learning"""
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

    return signals_1, signals_2

def pretrain_joint_flow_contrastive(model, signals, epochs, lr, logger):
    """Pretrain encoder using joint flow matching + contrastive learning"""
    logger.info(f"Starting joint flow + contrastive pretraining for {epochs} epochs...")

    # Initialize flow model
    target_channels = WINDOW_SIZE * N_CHANNELS  # Flattened signal dimensions
    z_channels = model.backbone.feature_dim     # Encoder output dimension

    using_fallback_flow = False
    flow_weight = 1.0
    contrastive_weight = 0.3

    if FLOW_AVAILABLE:
        try:
            flow_model = FlowLoss(
                target_channels=target_channels,
                z_channels=z_channels,
                depth=4,
                width=256,
                num_sampling_steps=20
            ).to(device)
            logger.info("✅ Using FlowLoss from PHM-Vibench")
        except Exception as e:
            logger.warning(f"FlowLoss initialization failed: {e}")
            flow_model = SimpleFlowModel(target_channels, z_channels).to(device)
            logger.info("Using SimpleFlowModel fallback")
            using_fallback_flow = True
    else:
        flow_model = SimpleFlowModel(target_channels, z_channels).to(device)
        logger.info("Using SimpleFlowModel (FlowLoss not available)")
        using_fallback_flow = True

    if using_fallback_flow:
        flow_weight = 0.1
        logger.info("Reducing flow loss weight to 0.1 and detaching encoder features for flow branch when using fallback model")

    # Setup optimizer for both encoder and flow model
    all_params = list(model.backbone.parameters()) + list(flow_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(signals))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    losses = []
    flow_losses = []
    contrastive_losses = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        flow_model.train()

        epoch_total_losses = []
        epoch_flow_losses = []
        epoch_contrastive_losses = []

        for batch_idx, (batch_signals,) in enumerate(dataloader):
            batch_signals = batch_signals.numpy()

            # Create augmented pairs for contrastive learning
            signals_1, signals_2 = create_augmented_pairs(batch_signals,
                                                        logger if batch_idx == 0 and epoch % 10 == 0 else None)

            # Convert to tensors and move to device
            signals_1 = torch.FloatTensor(signals_1).to(device)
            signals_2 = torch.FloatTensor(signals_2).to(device)

            # ==================== FLOW MATCHING LOSS ====================
            # Use original signals for flow matching
            target_signal = signals_1

            # Get conditional representation (without projection for flow)
            if hasattr(model.backbone, 'get_rep'):
                condition = model.backbone.get_rep(target_signal)
            else:
                _, condition = model.backbone(target_signal, use_projection=False)

            condition_for_flow = condition.detach() if using_fallback_flow else condition

            # Flatten signal for flow matching
            target_flat = target_signal.view(target_signal.size(0), -1)

            # Compute flow loss
            if FLOW_AVAILABLE and hasattr(flow_model, 'forward') and not using_fallback_flow:
                try:
                    flow_loss = flow_model(target_flat, condition_for_flow)
                except Exception as e:
                    logger.warning(f"FlowLoss forward failed: {e}, using fallback")
                    flow_loss = flow_model(target_flat, condition_for_flow)
            else:
                flow_loss = flow_model(target_flat, condition_for_flow)

            # ==================== CONTRASTIVE LOSS ====================
            # Get embeddings with projection heads for contrastive learning
            embeddings_1, _ = model.backbone(signals_1, use_projection=True)
            embeddings_2, _ = model.backbone(signals_2, use_projection=True)

            # Compute contrastive loss
            contr_loss = contrastive_loss(embeddings_1, embeddings_2, temperature=0.1)

            # ==================== COMBINED LOSS ====================
            # Joint loss: weighted combination of flow and contrastive objectives
            total_loss = flow_weight * flow_loss + contrastive_weight * contr_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()

            # Record losses
            epoch_total_losses.append(total_loss.item())
            epoch_flow_losses.append(flow_loss.item())
            epoch_contrastive_losses.append(contr_loss.item())

        # Update learning rate
        scheduler.step()

        # Calculate epoch metrics
        avg_total_loss = np.mean(epoch_total_losses)
        avg_flow_loss = np.mean(epoch_flow_losses)
        avg_contrastive_loss = np.mean(epoch_contrastive_losses)

        losses.append(avg_total_loss)
        flow_losses.append(avg_flow_loss)
        contrastive_losses.append(avg_contrastive_loss)

        # Early stopping check
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'flow_model_state_dict': flow_model.state_dict()
            }, 'checkpoints/best_joint_model.pth')
        else:
            patience_counter += 1

        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch:3d}/{epochs}: "
                       f"Total {avg_total_loss:.4f} "
                       f"(Flow {avg_flow_loss:.4f} + "
                       f"0.3×Contr {avg_contrastive_loss:.4f}), "
                       f"LR {current_lr:.2e}, "
                       f"Patience {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
            break

    # Load best model
    if os.path.exists('checkpoints/best_joint_model.pth'):
        checkpoint = torch.load('checkpoints/best_joint_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
        logger.info("Loaded best joint model")

    logger.info(f"Joint pretraining completed.")
    logger.info(f"Final losses - Total: {losses[-1]:.4f}, Flow: {flow_losses[-1]:.4f}, Contrastive: {contrastive_losses[-1]:.4f}")

    return {
        'total_losses': losses,
        'flow_losses': flow_losses,
        'contrastive_losses': contrastive_losses,
        'flow_model': flow_model,
        'flow_weight': flow_weight,
        'contrastive_weight': contrastive_weight,
        'using_flowloss': not using_fallback_flow
    }

def main():
    """Main execution function for Case 3"""
    start_time = time.time()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/case3_flow_contrastive_{timestamp}.log"
    logger = setup_logger("Case3_FlowContrastive", log_file)

    logger.info("="*80)
    logger.info("CASE 3: FLOW + CONTRASTIVE JOINT TRAINING + FEW-SHOT LEARNING")
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
        'case': 'Case 3 - Flow + Contrastive Joint Training + Few-Shot',
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
            'n_classes_anom': N_CLASSES_ANOM,
            'flow_weight': 1.0,
            'contrastive_weight': 0.3
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

        # ==================== PHASE 1: JOINT PRETRAINING ====================
        logger.info("\n" + "-"*60)
        logger.info("STEP 3: JOINT FLOW + CONTRASTIVE PRETRAINING PHASE")
        logger.info("-"*60)

        # Perform joint pretraining
        pretrain_results = pretrain_joint_flow_contrastive(
            model, signals, PRETRAIN_EPOCHS, LEARNING_RATE, logger
        )

        # Store pretraining results
        results['pretraining'] = {
            'total_losses': pretrain_results['total_losses'],
            'flow_losses': pretrain_results['flow_losses'],
            'contrastive_losses': pretrain_results['contrastive_losses'],
            'final_total_loss': pretrain_results['total_losses'][-1],
            'final_flow_loss': pretrain_results['flow_losses'][-1],
            'final_contrastive_loss': pretrain_results['contrastive_losses'][-1],
            'epochs_trained': len(pretrain_results['total_losses']),
            'num_samples': len(signals),
            'flow_available': FLOW_AVAILABLE,
            'using_flowloss': pretrain_results['using_flowloss'],
            'flow_weight': pretrain_results['flow_weight'],
            'contrastive_weight': pretrain_results['contrastive_weight']
        }

        results['hyperparameters']['flow_weight'] = pretrain_results['flow_weight']
        results['hyperparameters']['contrastive_weight'] = pretrain_results['contrastive_weight']

        logger.info(f"✅ Joint pretraining completed")
        logger.info(f"   Final total loss: {pretrain_results['total_losses'][-1]:.4f}")
        logger.info(f"   Final flow loss: {pretrain_results['flow_losses'][-1]:.4f}")
        logger.info(f"   Final contrastive loss: {pretrain_results['contrastive_losses'][-1]:.4f}")
        logger.info(f"   Epochs trained: {len(pretrain_results['total_losses'])}")

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
                'fine_tuning_method': 'unfrozen_adaptive_lr_after_joint_pretraining'
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
                'fine_tuning_method': 'unfrozen_adaptive_lr_after_joint_pretraining'
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
                    'fine_tuning_method': 'unfrozen_adaptive_lr_after_joint_pretraining'
                }

                logger.info(f"✅ Prediction task completed - Final MSE: {mse_values[-1]:.6f}")

        # ==================== RESULTS SUMMARY ====================
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time

        logger.info("\n" + "="*80)
        logger.info("CASE 3 RESULTS SUMMARY")
        logger.info("="*80)

        # Pretraining results
        logger.info(f"🌊 Joint Pretraining:")
        logger.info(f"   Total loss: {results['pretraining']['final_total_loss']:.4f}")
        logger.info(f"   Flow loss: {results['pretraining']['final_flow_loss']:.4f}")
        logger.info(f"   Contrastive loss: {results['pretraining']['final_contrastive_loss']:.4f}")
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
        results_file = f"results/case3_results_{timestamp}.pkl"
        save_results(results, results_file, logger)

        logger.info(f"\n✅ Case 3 completed successfully!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📝 Log saved to: {log_file}")

        return results

    except Exception as e:
        logger.error(f"❌ Case 3 failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise e

if __name__ == "__main__":
    results = main()

    # Print final summary to console
    print("\n" + "="*80)
    print("CASE 3 EXECUTION COMPLETED")
    print("="*80)

    print(f"Pretraining - Total: {results['pretraining']['final_total_loss']:.4f}, "
          f"Flow: {results['pretraining']['final_flow_loss']:.4f}, "
          f"Contrastive: {results['pretraining']['final_contrastive_loss']:.4f}")

    for task_name, task_results in results['tasks'].items():
        if task_name in ['diagnosis', 'anomaly']:
            acc = task_results['final_accuracy']
            print(f"{task_name.title():20s}: {acc:.4f} accuracy")
        elif task_name == 'prediction':
            mse = task_results['final_mse']
            print(f"{task_name.title():20s}: {mse:.6f} MSE")

    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print("Check logs/ and results/ directories for detailed output")
