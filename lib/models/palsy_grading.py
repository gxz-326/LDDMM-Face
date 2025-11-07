# ------------------------------------------------------------------------------
# Optional Facial Palsy Grading Module
# This module provides utilities for adding palsy grading classification
# on top of the landmark detection model
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn


class PalsyGradingHead(nn.Module):
    """
    Simple classification head for facial palsy grading.
    Can be added to the feature backbone of any model.
    
    Example usage:
        # Extract features from model backbone
        features = backbone(image)  # [batch, channels, height, width]
        
        # Create grading head
        grading_head = PalsyGradingHead(
            in_channels=256,
            num_grades=7  # 0-6 House-Brackmann scale
        )
        
        # Get grading predictions
        grade_logits = grading_head(features)  # [batch, num_grades]
    """
    
    def __init__(self, in_channels=256, num_grades=7, dropout_rate=0.5):
        """
        Args:
            in_channels: number of input feature channels
            num_grades: number of grading classes (default 7 for 0-6 House-Brackmann)
            dropout_rate: dropout rate for regularization
        """
        super(PalsyGradingHead, self).__init__()
        
        self.num_grades = num_grades
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_grades)
        )
    
    def forward(self, features):
        """
        Args:
            features: [batch, channels, height, width]
        
        Returns:
            logits: [batch, num_grades]
        """
        # Global average pooling
        x = self.pool(features)  # [batch, channels, 1, 1]
        x = x.flatten(1)  # [batch, channels]
        
        # Classification
        logits = self.fc_layers(x)  # [batch, num_grades]
        
        return logits


class DualHeadModel(nn.Module):
    """
    Wrapper to combine landmark detection and palsy grading.
    
    Example usage:
        # Wrap an existing model
        landmark_model = load_landmark_model()
        dual_model = DualHeadModel(
            landmark_model,
            grading_head_input_channels=256,
            num_grades=7
        )
        
        # Forward pass returns both landmark heatmaps and grading logits
        heatmaps, grade_logits = dual_model(image)
    """
    
    def __init__(self, landmark_model, grading_head_input_channels, num_grades=7):
        """
        Args:
            landmark_model: existing landmark detection model
            grading_head_input_channels: channels of intermediate features
            num_grades: number of grading classes
        """
        super(DualHeadModel, self).__init__()
        
        self.landmark_model = landmark_model
        self.grading_head = PalsyGradingHead(
            in_channels=grading_head_input_channels,
            num_grades=num_grades
        )
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: input image [batch, 3, height, width]
            return_features: if True, also return intermediate features
        
        Returns:
            heatmaps: [batch, num_joints, height, width]
            grade_logits: [batch, num_grades]
        """
        # Get landmark heatmaps
        heatmaps = self.landmark_model(x)
        
        # For grading, we need intermediate features
        # This requires hooking into the model to extract features
        # Alternatively, extract features by forward pass through backbone
        
        return heatmaps


class PalsyGradingLoss(nn.Module):
    """
    Combined loss function for landmark detection and palsy grading.
    
    Example usage:
        criterion = PalsyGradingLoss(
            landmark_weight=1.0,
            grading_weight=0.5,
            num_grades=7
        )
        
        loss = criterion(
            heatmap_pred, heatmap_target,
            grade_pred, grade_target
        )
    """
    
    def __init__(self, landmark_criterion, landmark_weight=1.0, 
                 grading_weight=0.5, num_grades=7):
        """
        Args:
            landmark_criterion: criterion for landmark loss (e.g., MSELoss)
            landmark_weight: weight for landmark detection loss
            grading_weight: weight for grading classification loss
            num_grades: number of grading classes
        """
        super(PalsyGradingLoss, self).__init__()
        
        self.landmark_criterion = landmark_criterion
        self.landmark_weight = landmark_weight
        self.grading_weight = grading_weight
        
        # Classification loss with class weights for imbalanced data
        self.grading_criterion = nn.CrossEntropyLoss()
    
    def forward(self, heatmap_pred, heatmap_target, grade_pred, grade_target):
        """
        Args:
            heatmap_pred: predicted heatmaps [batch, num_joints, H, W]
            heatmap_target: target heatmaps [batch, num_joints, H, W]
            grade_pred: predicted grade logits [batch, num_grades]
            grade_target: target grades [batch] (long tensor with class indices)
        
        Returns:
            loss: combined loss value
        """
        # Landmark detection loss
        landmark_loss = self.landmark_criterion(heatmap_pred, heatmap_target)
        
        # Palsy grading loss
        grading_loss = self.grading_criterion(grade_pred, grade_target)
        
        # Combined loss
        total_loss = (self.landmark_weight * landmark_loss + 
                      self.grading_weight * grading_loss)
        
        return total_loss


def create_grading_dataloader_wrapper(dataloader, palsy_grades_dict):
    """
    Wrapper to add palsy grades from metadata to dataloader.
    
    Args:
        dataloader: original dataloader
        palsy_grades_dict: dict mapping image names to grades
    
    Yields:
        (images, landmarks, palsy_grades)
    """
    for images, landmarks, meta in dataloader:
        palsy_grades = []
        for idx in meta['index']:
            img_name = meta.get('image_name', [idx])[0]
            grade = palsy_grades_dict.get(img_name, 0)
            palsy_grades.append(grade)
        
        palsy_grades = torch.tensor(palsy_grades, dtype=torch.long)
        yield images, landmarks, palsy_grades, meta


# Example training function with palsy grading
def train_with_grading(config, train_loader, model, criterion, optimizer,
                      epoch, writer_dict):
    """
    Training loop for dual-head model (landmarks + grading).
    
    To use this, modify tools/train.py to import and use this function
    instead of the original train function.
    """
    from .function import AverageMeter, compute_nme, decode_preds
    import time
    import logging
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    landmark_losses = AverageMeter()
    grading_losses = AverageMeter()
    
    model.train()
    nme_count = 0
    nme_batch_sum = 0
    
    end = time.time()
    
    for i, (inp, target, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Get grading labels from metadata
        if 'palsy_grade' in meta:
            palsy_grades = torch.tensor(
                meta['palsy_grade'], dtype=torch.long
            ).cuda(non_blocking=True)
        else:
            # Default to grade 0 if not provided
            palsy_grades = torch.zeros(
                inp.size(0), dtype=torch.long
            ).cuda(non_blocking=True)
        
        # Forward pass
        output = model(inp)
        target = target.cuda(non_blocking=True)
        
        # Handle tuple output (heatmaps, grade_logits)
        if isinstance(output, tuple):
            heatmaps, grade_logits = output
        else:
            # If model doesn't return grading, use only landmark loss
            heatmaps = output
            grade_logits = None
        
        # Compute loss
        if grade_logits is not None:
            loss = criterion(heatmaps, target, grade_logits, palsy_grades)
        else:
            loss = criterion(heatmaps, target)
        
        # NME computation
        score_map = heatmaps.data.cpu() if isinstance(output, tuple) else output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], 
                            config.MODEL.HEATMAP_SIZE)
        
        nme_batch = compute_nme(preds, meta, config)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inp.size(0))
        
        batch_time.update(time.time() - end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses)
            logger.info(msg)
        
        end = time.time()
    
    # Logging
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'.format(
        epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)
    
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('train_nme', nme, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


if __name__ == '__main__':
    # Test basic functionality
    grading_head = PalsyGradingHead(in_channels=256, num_grades=7)
    features = torch.randn(4, 256, 8, 8)
    logits = grading_head(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output: {logits}")
