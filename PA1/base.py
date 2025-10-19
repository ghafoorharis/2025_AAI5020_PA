import os
from turtle import mode
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from model import NetVLADModel
from train import train, validate
from eval import evaluate_recall


# Additonal Imports
try:
    import os
    import argparse
    import matplotlib.pyplot as plt
    from PIL import Image
    flag = True
except ImportError:
    print("Optional libraries for additional imports are not installed. Please install them using: pip install argparse matplotlib pillow")
    flag = False

class LoadDataset(Dataset):
    def __init__(self, query_file, db_file, root_dir, gt_npz, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        with open(query_file, 'r') as f:
            self.query_images = [os.path.join(root_dir, line.strip()) for line in f]
        with open(db_file, 'r') as f:
            self.db_images = [os.path.join(root_dir, line.strip()) for line in f]

        gt = np.load(gt_npz, allow_pickle=True)
        self.utmQ = gt["utmQ"]
        self.utmDb = gt["utmDb"]
        self.posDistThr = float(gt["posDistThr"])

    def __len__(self): return len(self.query_images)

    def __getitem__(self, idx):
        q_path = self.query_images[idx]
        q_img = Image.open(q_path).convert("RGB")

        dists = np.linalg.norm(self.utmDb - self.utmQ[idx], axis=1)

        pos_indices = np.where(dists < self.posDistThr)[0]
        if len(pos_indices) == 0: pos_indices = [np.argmin(dists)]
        p_path = self.db_images[np.random.choice(pos_indices)]
        p_img = Image.open(p_path).convert("RGB")

        neg_indices = np.where(dists > 10 * self.posDistThr)[0]
        if len(neg_indices) == 0: neg_indices = [np.argmax(dists)]
        n_path = self.db_images[np.random.choice(neg_indices)]
        n_img = Image.open(n_path).convert("RGB")

        if self.transform:
            q_img = self.transform(q_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        return q_img, p_img, n_img

class ImageListDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        with open(list_file, "r") as f:
            self.paths = [os.path.join(root_dir, line.strip()) for line in f]
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, idx, p

def main(args):
    os.makedirs(f"{args.checkpoint_dir}", exist_ok=True)
    os.makedirs(f"{args.save_dir}", exist_ok=True)

    EPOCHS = args.epochs
    # LR = args.lr
    # BATCH_SIZE = args.batch_size
    # NUM_WORKERS = args.num_workers
    # DEVICE = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = LoadDataset(
        query_file="dataset/query_train.txt",
        db_file="dataset/index_train.txt",
        root_dir="",
        gt_npz="dataset/gt/pitts30k_train.npz",
        transform=transform
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    # -------------------------------
    # TASK1 : VGG16-based NetVLAD Structure
    # -------------------------------
    model = NetVLADModel(num_clusters=16).to(device)
    print(model)

    # -------------------------------
    # TASK2 : Learning with Ranking Loss
    # -------------------------------
    os.makedirs(f"{args.checkpoint_dir}", exist_ok=True)
    last_epoch_model_path = f"{args.checkpoint_dir}/netvlad_final.pth"
    if not args.train:
        if os.path.exists(last_epoch_model_path):
            model.load_state_dict(torch.load(last_epoch_model_path))
            print(f"Loaded model from {last_epoch_model_path}")
        else:
            print(f"No model found at {last_epoch_model_path}")
    else:
        train(model, train_loader, val_loader, device, epochs=EPOCHS,args=args)
        torch.save(model.state_dict(), f"{args.checkpoint_dir}/netvlad_final.pth")
    # else:
    #     print("No training")
    #     print("Loading model from checkpoint")
    #     model.load_state_dict(torch.load(last_epoch_model_path))
    test_loss = validate(model, test_loader, device,args=args)
    print(f"Final Test Loss = {test_loss:.4f}")
    # # -------------------------------
    # # TASK3 : Evaluation with Recall@K
    # # -------------------------------
    gt = np.load("dataset/gt/pitts30k_test.npz", allow_pickle=True)
    query_ds = ImageListDataset("dataset/query_test.txt", "", transform)
    db_ds = ImageListDataset("dataset/index_test.txt", "", transform)
    # evaluate_recall(model, query_ds, db_ds, gt["utmQ"], gt["utmDb"], float(gt["posDistThr"]), device, recall_values=[1,5,10])
    # Evaluate with visualization
    recalls, sim_scores, ground_truth_dist = evaluate_recall(
        model=model,
        query_ds=query_ds,
        db_ds=db_ds, 
        utmQ=gt["utmQ"],
        utmDb=gt["utmDb"],
        posDistThr=gt["posDistThr"],
        device=device,
        recall_values=[1, 5, 10,15,20,25,30],
        preview=args.preview,  # Set to True to generate visualizations
        save_dir=args.save_dir,  # Directory to save figures
        num_examples=len(query_ds),
        )
    with open("output/recall_results.txt", "w") as f:
        f.write(f"Recall@K RESULTS\n")
        f.write(f"=================\n")
        for k in recalls:
            f.write(f"Recall@{k}: {recalls[k]:.2f}%\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train", type= bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="./output/checkpoints")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--preview", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default="./output/visualizations")
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
