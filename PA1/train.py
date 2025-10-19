import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def validate(model, loader, device,args  = None):
    """
    Your Task) Complete the function in train.py.

    - Define a margin-based ranking loss that encourages query vectors
      to be closer to positive samples than negative ones.
    - Use the same loss formulation as in training:
          loss = max(0, d(q,p) + margin - d(q,n))
    - Compute the average validation loss across all batches.

    Returns:
        float: Average validation loss.
    """
    criterion = nn.TripletMarginLoss(args.margin)
    total_loss = 0.0

    # TODO:
    # 1. Set the model to evaluation mode.
    model.eval()
    # 2. Disable gradient computation (torch.no_grad).
    with torch.no_grad():
        # 3. For each batch, extract embeddings for q, p, n.
        for q, p, n in tqdm(loader, desc="Validation"):
            q, p, n = q.to(device), p.to(device), n.to(device)
            q_emb, p_emb, n_emb = model(q), model(p), model(n)
            # 4. Apply the triplet margin loss.
            loss = criterion(q_emb, p_emb, n_emb)
            # 5. Accumulate the loss and return the average.
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    args = None,
):
    """
    Your Task) Complete the function in train.py.

    - Implement the training loop with a margin-based triplet ranking loss.
    - Loss formulation:
          loss = max(0, d(q,p) + margin - d(q,n))
    - Optimize the model using Adam optimizer.
    - After each epoch:
        * Print the average training loss.
        * Evaluate on the validation set using `validate`.
    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        device: The device to train on.
        epochs: The number of epochs to train for.
        lr: The learning rate to use.
        args: The arguments to use.
    Returns:
        None
    """
    # TODO:
    # 1. Define a nn.TripletMarginLoss.
    criterion = nn.TripletMarginLoss(margin=args.margin)
    # 2. Define an optim.Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 3. For each epoch:
    for epoch in range(epochs):
        # - Set model to train mode.
        model.train()
        total_loss = 0.0
        # 4. Loop over training batches:
        for q, p, n in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} "):
            q, p, n = q.to(device), p.to(device), n.to(device)
            # 5. Compute embeddings for q, p, n.
            q_emb, p_emb, n_emb = model(q), model(p), model(n)
            # 6. Compute the triplet margin loss.
            loss = criterion(q_emb, p_emb, n_emb)
            # 7. Zero the gradients to prevent accumulation
            optimizer.zero_grad()
            # 8. Backpropagate the loss.
            loss.backward()
            # 9. Update the model parameters.
            optimizer.step()
            total_loss += loss.item()

        # 10. Compute the average training loss.
        avg_loss = total_loss / len(train_loader)
        # 11. Print the training loss.
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")
        # 12. Evaluate on the validation set using `validate`.
        val_loss = validate(model, val_loader, device,criterion)
        # 13. Print the validation loss.
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
        # log the training and validation loss
        with open("train.log", "a") as f:
            f.write(
                f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}\n"
            )
        torch.save(model.state_dict(), f"{args.checkpoint_dir}/netvlad_final.pth")
