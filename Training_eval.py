from tqdm import tqdm 
import torch
from metrics import compute_pesq, compute_sdr
from InvertMelSpectrogram import MelToWaveform

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader,total=len(train_loader), desc="Training"):
        optimizer.zero_grad()
        audio, target = batch
        audio = audio.to(device)
        target = target.to(device)
        output = model(audio, target)
        output = output.unsqueeze(1)  # Ensure output is [B, T]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, compute_metrics=True):
    model.eval()
    total_loss = 0
    sdr_total = 0.0
    pesq_total = 0.0
    n_batches = 0
    mel_to_waveform = MelToWaveform().to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc="Evaluating"):
            audio, target = batch
            audio = (audio).to(device)
            target =(target).to(device)

            output = model(audio, target)
            output = output.unsqueeze(1) 
            loss = criterion(output, target)
            total_loss += loss.item()

            if compute_metrics:
                reconstructed_waveform = mel_to_waveform(output)  # Shape: [B, T]

                # If model output is not [B, T], make sure to squeeze
                pred_audio = output.squeeze()
                clean_audio = target.squeeze()

                # If mono, ensure [B, T] shape
                if pred_audio.ndim == 1:
                    pred_audio = pred_audio.unsqueeze(0)
                    clean_audio = clean_audio.unsqueeze(0)

                sdr_score = compute_sdr(clean_audio, pred_audio).item()
                pesq_score = compute_pesq(clean_audio, pred_audio, sr=16000)

                sdr_total += sdr_score
                pesq_total += pesq_score
                n_batches += 1

    avg_loss = total_loss / len(val_loader)
    avg_sdr = sdr_total / n_batches if compute_metrics else None
    avg_pesq = pesq_total / n_batches if compute_metrics else None

    return {
        "loss": avg_loss,
        "sdr": avg_sdr,
        "pesq": avg_pesq
    }