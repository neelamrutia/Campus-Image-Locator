# GeoCoord Location Predictor

1. **Setup & Reproducibility**  
   - Seed NumPy and PyTorch for consistent results; use GPU if available.  
   - Train for 100 epochs with mixed-precision (`GradScaler`) and cosine LR scheduling over 30 steps.  

2. **Data Loading & Preprocessing**  
   - Read CSVs of filenames, latitude, and longitude; filter out-of-range coordinates and a few bad validation IDs.  
   - Compute mean and standard deviation on train coords for normalization.  
   - Images: resize, random crop + RandAugment + ColorJitter for train; center-crop for val; normalize to ImageNet stats.  

3. **Model Architecture**  
   - Pretrained DINOv2 ViT-B/14 backbone with early layers frozen and DropPath on later blocks.  
   - Regression head: 768 → 384 → 2 with dropout and ReLU, outputting normalized coordinate deltas.  
   - for better results ran the multiple models and use the ensemble methods to get better results.

4. **Losses & Metrics**  
   - Train with Smooth L1 loss on normalized coords.  
   - Compute MSE in original scale per latitude and longitude by denormalizing predictions.  

5. **Training & Validation**  
   - AdamW optimizer with low LR for backbone (2e-5) and higher for head (2e-3), weight decay 1e-2.  
   - Clip gradients and step scheduler each epoch.  
   - On validation, log normalized loss and denormalized MSE for lat, lon, and average.  
   - Save model checkpoint whenever average MSE improves; final best MSE is printed.

Drive Link For model - https://drive.google.com/drive/folders/1-hiPlFNjWcBppvNyj8F61VJdinKyGYSf?usp=share_link

