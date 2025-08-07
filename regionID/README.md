# Region Classification Model

1. **Setup & Device**  
   - Uses GPU if available; otherwise CPU.  
   - Main script guarded by `if __name__ == "__main__"` for direct execution.

2. **Data Loading**  
   - CSV files list image filenames and `Region_ID` labels (1–15).  
   - Custom `RegionDataset` returns `(image, label-1, filename)` for 0-based labels.  

3. **Transforms**  
   - **Train**: Resize to 256×256 → RandomResizedCrop(224) → Horizontal flip → Color jitter → Normalize to ImageNet stats.  
   - **Val**: Resize to 224×224 → Normalize to ImageNet stats.  

4. **DataLoaders**  
   - Batch size 32, 4 workers.  
   - Shuffle enabled for training, disabled for validation.  

5. **Model Builder**  
   - Supports `efficientnet_b0`, `resnet50`, `convnext_tiny`, `mobilenet_v3_large`.  
   - Loads pretrained weights, replaces final classifier to output 15 classes.  

6. **Loss & Metrics**  
   - Cross-entropy loss for 15-way classification.  
   - Accuracy computed via `sklearn.metrics.accuracy_score`.  

7. **Optimizer & Scheduler**  
   - AdamW optimizer (LR 3e-4, weight decay 1e-5).  
   - CosineAnnealingLR over the total number of epochs, min LR 1e-6.  

8. **Training Loop**  
   - Per-epoch: train phase logs loss & accuracy; validation phase logs loss & accuracy.  
   - `tqdm` progress bars for both loops.  

9. **Checkpointing**  
   - Saves best model weights to `best_<model_name>.pth`.  
   - Exports a CSV of filenames with predicted `Region_ID` (1–15).  

10. **Execution**  
   - Default choice: `convnext_tiny` for 100 epochs.  
   - Prints final best validation accuracy and saved model path.  


- Drive Link For model - https://drive.google.com/drive/folders/13OQIQomCX90eLZGvIMGgipY-kyCb_uUp?usp=share_link

