import os
import torch
from torch.utils.data import DataLoader
# Importing custom modular assets cleanly from sibling files
from model_architecture import TransUNetHybrid
from utils import HybridLoss, GlacialLakeDataset

def run_training_session():
    # --- CONFIGURABLE DIRECTORIES ---
    IMAGE_DIR = "./dataset/train_images"   #give image directory here 
    MASK_DIR = "./dataset/train_masks"   #give mask directory here 
    SAVE_PATH = "./best_transunet_model.pt"   #model destination here
    
    EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training engine executing on device framework: {device}")

    # Initialize data management pipelines
    train_dataset = GlacialLakeDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Instantiate structural networks
    model = TransUNetHybrid().to(device)
    criterion = HybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(" Training optimization loops successfully running...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"| Epoch [{epoch+1:02d}/{EPOCHS:02d}] ──► Calculated Batch Matrix Loss: {epoch_loss:.4f}")

    # Save target parameters state dict
    torch.save(model.state_dict(), SAVE_PATH)
    print(f" Training optimized constraints written to storage: {SAVE_PATH}")

if __name__ == "__main__":
    # Protect sequence against nested processing exceptions
    if os.path.exists("./dataset/train_images"):
        run_training_session()
    else:
        print(" Execution script validated. Setup your path matrices to trigger full loop training.")
