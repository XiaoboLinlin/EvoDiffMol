"""
Training utilities for diffusion model training.
Extracted from train.py to make it reusable across different training scripts.
"""

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from datasets.qm9.utils import prepare_context


def train_epoch(model, train_loader, optimizer_global, optimizer_local, 
                config, device, writer=None, logger=None, iteration=None,
                property_norms=None, context_args=None, ema=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The diffusion model
        train_loader: Training data loader
        optimizer_global: Global optimizer
        optimizer_local: Local optimizer
        config: Training configuration
        device: Device for training
        writer: TensorBoard writer (optional)
        logger: Logger instance (optional)
        iteration: Current iteration number (optional)
        property_norms: Property normalization parameters (optional)
        context_args: Context arguments (optional)
        ema: Exponential moving average model (optional)
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    loss_vae_kl = 0.0
    
    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for batch in train_loader:
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            batch = batch.to(device)
            
            # Extract update_mask for scaffold training if available
            update_mask = None
            if hasattr(batch, 'update_mask'):
                update_mask = batch.update_mask
            
            # Prepare context if specified
            context = None
            if context_args and len(context_args) > 0 and property_norms is not None:
                context = prepare_context(context_args, batch, property_norms)

            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True,
                update_mask=update_mask
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss
            
            # Note: update_mask is now handled inside the model's forward method
            # Use the total loss already computed by the model
            loss = loss.mean()
            
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer_global.step()
            optimizer_local.step()
            
            # Update EMA if provided
            if ema is not None:
                ema.update()
            
            sum_loss += loss.item()
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()
            
            sum_n += 1
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)

    # Calculate averages
    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n

    # Log results
    if logger and iteration is not None:
        logger.info(
            f'[Train] Iter {iteration:05d} | Loss {avg_loss:.2f} | '
            f'Loss(pos_Global) {avg_loss_pos_global:.2f} | Loss(pos_Local) {avg_loss_pos_local:.2f} | '
            f'Loss(node_global) {avg_loss_node_global:.2f} | Loss(node_local) {avg_loss_node_local:.2f} | '
            f'Loss(vae_KL) {loss_vae_kl:.2f} | Grad {orig_grad_norm:.2f} | '
            f'LR {optimizer_global.param_groups[0]["lr"]:.6f}'
        )
    
    # Write to tensorboard
    if writer and iteration is not None:
        writer.add_scalar('train/loss', avg_loss, iteration)
        writer.add_scalar('train/loss_pos_global', avg_loss_pos_global, iteration)
        writer.add_scalar('train/loss_node_global', avg_loss_node_global, iteration)
        writer.add_scalar('train/loss_pos_local', avg_loss_pos_local, iteration)
        writer.add_scalar('train/loss_node_local', avg_loss_node_local, iteration)
        writer.add_scalar('train/loss_vae_KL', loss_vae_kl, iteration)
        writer.add_scalar('train/lr', optimizer_global.param_groups[0]['lr'], iteration)
        writer.add_scalar('train/grad_norm', orig_grad_norm, iteration)
        writer.flush()
    
    return {
        'avg_loss': avg_loss,
        'avg_loss_pos_global': avg_loss_pos_global,
        'avg_loss_node_global': avg_loss_node_global,
        'avg_loss_pos_local': avg_loss_pos_local,
        'avg_loss_node_local': avg_loss_node_local,
        'loss_vae_kl': loss_vae_kl,
        'grad_norm': orig_grad_norm,
        'lr': optimizer_global.param_groups[0]['lr']
    }


def validate_epoch(model, val_loader, config, device, scheduler_global=None, 
                   scheduler_local=None, writer=None, logger=None, iteration=None,
                   property_norms=None, context_args=None, ema=None):
    """
    Validate the model for one epoch.
    
    Args:
        model: The diffusion model
        val_loader: Validation data loader
        config: Training configuration
        device: Device for validation
        scheduler_global: Global scheduler (optional)
        scheduler_local: Local scheduler (optional)
        writer: TensorBoard writer (optional)
        logger: Logger instance (optional)
        iteration: Current iteration number (optional)
        property_norms: Property normalization parameters (optional)
        context_args: Context arguments (optional)
        ema: Exponential moving average model (optional)
    
    Returns:
        Average validation loss
    """
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    
    with torch.no_grad():
        # Apply EMA weights for validation if available
        if ema is not None:
            ema.apply_shadow()
        
        model.eval()
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            
            # Extract update_mask for scaffold training if available
            update_mask = None
            if hasattr(batch, 'update_mask'):
                update_mask = batch.update_mask
                
            # Prepare context if specified
            context = None
            if context_args and len(context_args) > 0 and property_norms is not None:
                context = prepare_context(context_args, batch, property_norms)
                
            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True,
                update_mask=update_mask
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss

            # Note: update_mask is now handled inside the model's forward method
            # Use the total loss already computed by the model
            
            sum_loss += loss.mean().item()
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()
            
            sum_n += 1
        
        # Restore original weights if using EMA
        if ema is not None:
            ema.restore()

    # Calculate averages
    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n

    # Update schedulers
    if scheduler_global is not None and scheduler_local is not None:
        if config.train.scheduler.type == 'plateau':
            scheduler_global.step(avg_loss_pos_global + avg_loss_node_global)
            scheduler_local.step(avg_loss_pos_local + avg_loss_node_local)
        else:
            scheduler_global.step()
            if 'global' not in config.model.network:
                scheduler_local.step()

    # Log results
    if logger and iteration is not None:
        logger.info('[Validate] Iter %05d | Loss %.6f' % (iteration, avg_loss))
    
    # Write to tensorboard
    if writer and iteration is not None:
        writer.add_scalar('val/loss', avg_loss, iteration)
        writer.flush()
    
    return avg_loss


def train_scaffold_fine_tuning(model, scaffold_data_with_masks, optimizer_global, 
                               optimizer_local, device, config, num_epochs=20, 
                               batch_size=32, log_prefix="Scaffold fine-tuning"):
    """
    Fine-tune the model on scaffold-containing molecules.
    
    Args:
        model: The diffusion model
        scaffold_data_with_masks: List of data objects with update masks
        optimizer_global: Global optimizer
        optimizer_local: Local optimizer 
        device: Device for training
        config: Training configuration
        num_epochs: Number of epochs to fine-tune
        batch_size: Batch size for fine-tuning
        log_prefix: Prefix for logging
    
    Returns:
        Dictionary with fine-tuning metrics
    """
    import logging
    from torch_geometric.data import DataLoader
    
    logger = logging.getLogger('genetic_training')
    logger.info(f"{log_prefix}: Starting {num_epochs} epochs of fine-tuning on {len(scaffold_data_with_masks)} molecules")
    
    # Create data loader for scaffold data
    train_loader = DataLoader(scaffold_data_with_masks, batch_size=batch_size, shuffle=True)
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        with tqdm(total=len(train_loader), desc=f'{log_prefix} Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in train_loader:
                optimizer_global.zero_grad()
                optimizer_local.zero_grad()
                batch = batch.to(device)
                
                # Extract update_mask
                update_mask = batch.update_mask
                
                loss = model(
                    batch,
                    context=None,
                    return_unreduced_loss=True,
                    update_mask=update_mask
                )

                if config.model.vae_context:
                    loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, _ = loss
                else:
                    loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss
                
                # Apply update_mask to losses
                if update_mask.dim() == 1:
                    update_mask = update_mask.unsqueeze(-1)
                
                loss_pos_global = loss_pos_global * update_mask
                loss_pos_local = loss_pos_local * update_mask
                loss_node_global = loss_node_global * update_mask
                loss_node_local = loss_node_local * update_mask
                loss = loss_pos_global + loss_pos_local + loss_node_global + loss_node_local
                
                # Compute mean only over unmasked atoms
                num_unmasked = update_mask.sum()
                loss_mean = loss.sum() / num_unmasked
                
                loss_mean.backward()
                clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer_global.step()
                optimizer_local.step()
                
                epoch_loss += loss_mean.item()
                num_batches += 1
                pbar.set_postfix({'loss': '%.4f' % loss_mean.item()})
                pbar.update(1)
        
        avg_epoch_loss = epoch_loss / num_batches
        final_loss = avg_epoch_loss
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"{log_prefix}: Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    logger.info(f"{log_prefix}: Completed! Final loss: {final_loss:.4f}")
    
    return {
        'final_loss': final_loss,
        'num_epochs': num_epochs,
        'num_molecules': len(scaffold_data_with_masks)
    }


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Store original parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                   (1.0 - self.decay) * param.data)
    
    def apply_shadow(self, model):
        """Apply shadow weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                param.data = self.original[name]
        self.original = {} 