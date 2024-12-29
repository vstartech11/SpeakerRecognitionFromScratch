import multiprocessing
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, roc_curve

import dataset
import feature_extraction
import myconfig


class BaseSpeakerEncoder(nn.Module):
    def _load_from(self, saved_model):
        var_dict = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        # Define the LSTM network.
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(
                batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]

    def forward(self, x):
        D = 2 if myconfig.BI_LSTM else 1
        h0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0],  myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)
        c0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(myconfig.N_MFCC, myconfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=myconfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=1)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, myconfig.TRANSFORMER_DIM).to(
            myconfig.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    if myconfig.USE_TRANSFORMER:
        return TransformerSpeakerEncoder(load_from).to(myconfig.DEVICE)
    else:
        return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)


def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0.0))


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = torch.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1]))
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :])
    loss = torch.mean(batch_loss)
    return loss


def save_model(saved_model_path, encoder, losses, start_time):
    """Save model to disk."""
    training_time = time.time() - start_time
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".pt"):
        saved_model_path += ".pt"
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "losses": losses,
                "training_time": training_time},saved_model_path)


def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    losses = []
    encoder = get_speaker_encoder()

    # Train
    optimizer = optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Build batched input.
        batch_input = feature_extraction.get_batched_triplet_input(
            spk_to_utts, myconfig.BATCH_SIZE, pool).to(myconfig.DEVICE)

        # Compute loss.
        batch_output = encoder(batch_input)
        loss = get_triplet_loss_from_batch_output(
            batch_output, myconfig.BATCH_SIZE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and
                (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint,
                       encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses

# Tambahkan fungsi evaluasi

def calculate_eer(fpr, tpr):
    """Calculate Equal Error Rate (EER)"""
    fnr = 1 - tpr
    eer_threshold = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold] + fnr[eer_threshold]) / 2
    return eer

def evaluate_model(encoder, test_data):
    """Evaluate the model and calculate EER and AUC."""
    encoder.eval()
    with torch.no_grad():
        positive_scores = []
        negative_scores = []

        for anchor, positive, negative in test_data:
            anchor_emb = encoder(anchor.to(myconfig.DEVICE))
            positive_emb = encoder(positive.to(myconfig.DEVICE))
            negative_emb = encoder(negative.to(myconfig.DEVICE))

            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            positive_scores.append(cos(anchor_emb, positive_emb).cpu().numpy())
            negative_scores.append(cos(anchor_emb, negative_emb).cpu().numpy())

        # Combine scores
        scores = np.concatenate([positive_scores, negative_scores])
        labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Calculate EER
        eer = calculate_eer(fpr, tpr)

    return eer, roc_auc, fpr, tpr


# Modifikasi fungsi run_training
def run_training():
    
    torch.manual_seed(myconfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myconfig.SEED)
    np.random.seed(myconfig.SEED)
    random.seed(myconfig.SEED)
    if myconfig.USE_CUSTOMIZE_DATASETS:
        spk_to_utts = dataset.get_customize_spk_to_utts(
            myconfig.TRAIN_DATA_DIR)
        print("Training data with customize datasets:", myconfig.TRAIN_DATA_DIR)
    else:
        spk_to_utts = dataset.get_customize_spk_to_utts(
            myconfig.TRAIN_DATA_DIR)
        print("Training data with Librispeech datasets:", myconfig.TRAIN_DATA_DIR)

    print(len(spk_to_utts), "speakers")
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts,
                               myconfig.TRAINING_STEPS,
                               myconfig.SAVED_MODEL_PATH,
                               pool)

    # Menampilkan grafik menggunakan Matplotlib
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss")

    # Simpan grafik sebagai file gambar dengan nama tanggal dan waktu
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)  # Pastikan direktori output ada
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"training_loss_plot_{timestamp}_{myconfig.TRAINING_STEPS}steps_{myconfig.LEARNING_RATE}lr.png")
    plt.savefig(output_file)
    print(f"Grafik loss telah disimpan di: {output_file}")

    # Tampilkan grafik
    plt.show()


def menu():
    """Display menu for selecting train or evaluation."""
    print("Pilih mode operasi:")
    print("1. Train")
    print("2. Evaluasi")
    choice = input("Masukkan pilihan (1/2): ")
    return choice.strip()

def main():
    choice = menu()
    if choice == "1":
        print("Mode: Train")
        run_training()
    elif choice == "2":
        print("Mode: Evaluasi")
        if myconfig.TEST_DATA_DIR:
            spk_to_utts = dataset.get_customize_spk_to_utts(
            myconfig.TEST_DATA_DIR)
            encoder = get_speaker_encoder(
        myconfig.SAVED_MODEL_PATH)

            eer, roc_auc, fpr, tpr = evaluate_model(encoder, spk_to_utts)

            print(f"EER: {eer * 100:.2f}%")
            print(f"AUC: {roc_auc:.4f}")

            # Visualize ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            plt.show()
        else:
            print("Error: Direktori data uji tidak ditemukan. Periksa konfigurasi.")
    else:
        print("Pilihan tidak valid. Harap masukkan 1 atau 2.")


if __name__ == "__main__":
    main()