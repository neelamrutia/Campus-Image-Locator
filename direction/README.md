1. **Setup & Reproducibility**  
   - Fix random seeds and use GPU if available.  
   - Mixed-precision training with GradScaler for speed.  
   - Cosine LR schedule over 100 epochs.  
   - Save the best model and predictions automatically.

2. **Data & Augmentation**  
   - Read image filenames and angles from CSV.  
   - Convert angles to [cos, sin] vectors.  
   - Train images: resize, random crop, RandAugment, ColorJitter, normalize.  
   - Validation: center-crop and normalize.  
   - Load data in small batches with 4 workers.

3. **Model**  
   - Pretrained DINOv2 ViT-B/14 backbone.  
   - Freeze early layers, add DropPath in later blocks.  
   - Simple head: 768 → 384 → 2 with dropout and ReLU.  
   - Output normalized to unit vector.
   - for better results ran the multiple models and use the ensemble methods to get better results.

4. **Training**  
   - MSE loss on vector outputs.  
   - Angular MAE metric.  
   - AdamW optimizer: low LR for backbone, higher for head.  
   - Gradient accumulation and clipping for stable updates.

5. **Validation & Saving**  
   - Compute loss and MAE on the validation set.  
   - Convert predictions back to angles (0–360°).  
   - Save best model and CSV of best predictions.  
   - Print final best MAE.  


Drive Link For model - https://drive.google.com/drive/folders/1T4m7FcXGEEIC__4UJXwwKuxApamzOsot?usp=share_link

