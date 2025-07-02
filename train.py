import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from network import MobileMatting
from dataset import Composition1KDataset, get_train_transforms, get_val_transforms, collate_fn
from losses import MattingLoss, compute_sad, compute_mse, compute_gradient_error


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = MobileMatting().to(self.device)
        
        # Create loss function
        self.criterion = MattingLoss(
            alpha_weight=config.alpha_weight,
            comp_weight=config.comp_weight,
            grad_weight=config.grad_weight,
            lap_weight=config.lap_weight
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Create datasets
        train_transform = get_train_transforms(config.input_size)
        val_transform = get_val_transforms(config.input_size)
        
        self.train_dataset = Composition1KDataset(
            config.data_root,
            mode='train',
            transform=train_transform,
            input_size=config.input_size
        )
        
        self.val_dataset = Composition1KDataset(
            config.data_root,
            mode='test',
            transform=val_transform,
            input_size=config.input_size
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Create tensorboard writer
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'alpha': 0.0,
            'composition': 0.0,
            'gradient': 0.0,
            'laplacian': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch['image'].to(self.device)
            alpha_gt = batch['alpha'].to(self.device)
            trimap = batch['trimap'].to(self.device)
            clicks = batch['clicks'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            alpha_pred = self.model(image, trimap, clicks)
            
            # Compute loss
            losses = self.criterion(alpha_pred, alpha_gt, image, trimap)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'Train/{key}_loss', value.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Alpha': f"{losses['alpha'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.global_step += 1
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'alpha': 0.0,
            'composition': 0.0,
            'gradient': 0.0,
            'laplacian': 0.0
        }
        val_metrics = {
            'sad': 0.0,
            'mse': 0.0,
            'grad_error': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move to device
                image = batch['image'].to(self.device)
                alpha_gt = batch['alpha'].to(self.device)
                trimap = batch['trimap'].to(self.device)
                clicks = batch['clicks'].to(self.device)
                
                # Forward pass
                alpha_pred = self.model(image, trimap, clicks)
                
                # Compute loss
                losses = self.criterion(alpha_pred, alpha_gt, image, trimap)
                
                # Compute metrics
                sad = compute_sad(alpha_pred, alpha_gt, trimap)
                mse = compute_mse(alpha_pred, alpha_gt, trimap)
                grad_error = compute_gradient_error(alpha_pred, alpha_gt, trimap)
                
                # Update metrics
                for key in val_losses:
                    val_losses[key] += losses[key].item()
                
                val_metrics['sad'] += sad.item()
                val_metrics['mse'] += mse.item()
                val_metrics['grad_error'] += grad_error.item()
                
                pbar.set_postfix({
                    'Loss': f"{losses['total'].item():.4f}",
                    'SAD': f"{sad.item():.2f}",
                    'MSE': f"{mse.item():.4f}"
                })
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_losses, val_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {self.best_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses, val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            print(f"Val SAD: {val_metrics['sad']:.2f}")
            print(f"Val MSE: {val_metrics['mse']:.4f}")
            print(f"Val Grad Error: {val_metrics['grad_error']:.4f}")
            
            # Log to tensorboard
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}_loss', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_loss:.4f}")
        
        self.writer.close()


def get_config():
    """Get training configuration"""
    class Config:
        # Data
        data_root = "/content/Composition-1K"  # Update this path for your Colab setup
        input_size = 512
        
        # Training
        batch_size = 8
        val_batch_size = 4
        epochs = 100
        learning_rate = 1e-4
        weight_decay = 1e-4
        num_workers = 2
        
        # Loss weights
        alpha_weight = 1.0
        comp_weight = 1.0
        grad_weight = 1.0
        lap_weight = 1.0
        
        # Logging and saving
        log_interval = 100
        save_interval = 10
        checkpoint_dir = "/content/checkpoints"
        log_dir = "/content/logs"
        
        # Resume training
        resume_checkpoint = None
    
    return Config()


def main():
    # Get configuration
    config = get_config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        trainer.load_checkpoint(config.resume_checkpoint)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
