from Transformer import Transformer
import AudioDataset
from loss_fn import DynamicFourierLoss
from Training_eval import train, evaluate
from EarlyStopping import EarlyStopping
import os
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # ✅ fixed typo

def main(direc, batch_size=32, num_epochs=400):
    # Split dataset paths into train/val/test
    with open(direc, 'r') as f:
        lines = list(map(lambda x: x.strip("\n")[x.index(",")+1:], f.readlines()))[1:]
    train_data, val_test_data = train_test_split(lines, test_size=0.5)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5)

    # Create dataloaders
    train_loader = AudioDataset.create_dataloader(train_data, batch_size)
    val_loader = AudioDataset.create_dataloader(val_data, batch_size)
    test_loader = AudioDataset.create_dataloader(test_data, batch_size)

    # Initialize model
    model = Transformer(

        input_dim=126, output_dim=126,
        embed_dim=256, num_heads=4,
        num_encoder_layers=4, num_decoder_layers=4,
        ff_dim=512, dropout=0.1
    )

    # Loss, optimizer, device
    loss_fn = DynamicFourierLoss(alpha_init=1.0, smoothing=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Early stopping
    early_stopping = EarlyStopping(patience=20, delta=1e-2)

    # Logs
    train_loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epochs"):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        train_loss_list.append(train_loss)
        val_metrics = evaluate(model, val_loader, loss_fn, device, compute_metrics=False)
        val_loss = val_metrics["loss"]
        val_loss_list.append(val_loss)

        print(val_metrics)      # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        else:
            print(f"Patience: {early_stopping.counter} | Best Loss: {early_stopping.best_score:.4f}")

    # Final test evaluation
    print("\nEvaluating on Test Set:")
    test_metrics = evaluate(model, test_loader, loss_fn, device, compute_metrics=True)
    print(f"Test Loss: {test_metrics['loss']:.4f} | SDR: {test_metrics['sdr']:.2f} dB | PESQ: {test_metrics['pesq']:.2f}")
    # Save model
    if not os.path.exists("transformer_model"):
        os.makedirs("transformer_model")
    torch.save(model.state_dict(), "./transformer_model/transformer_model.pth")


if __name__ == "__main__":
    # Example usage
    data_dir = "./files_list.csv"
    main(data_dir)