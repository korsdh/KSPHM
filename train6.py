import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from model2 import UnetRMSPredictor

# train_set = torch.load("cachedRMS\Train_Set_RMS.pt", weights_only=False)
# val_set   = torch.load("cachedRMS\Validation_Set_RMS.pt",   weights_only=False)
# test_set  = torch.load("cachedRMS\Test_Set_RMS.pt",  weights_only=False)

# train_df = pd.DataFrame(train_set)
# val_df = pd.DataFrame(val_set)
# test_df = pd.DataFrame(test_set)

def load_rms_cache(pt_path):
    return torch.load(pt_path)

# class RMSSimpleDataset(Dataset):
#     def __init__(self, cached_list, window_size=5, output_len=1, stride=1):
#         """
#         cached_list: torch.load()로 불러온 리스트
#         window_size: 과거 몇 개의 RMS값을 입력으로 볼지
#         output_len: 예측할 RMS 개수 (여기서는 1)
#         stride: 시퀀스 슬라이딩 간격
#         """
#         self.window_size = window_size
#         self.output_len = output_len
#         self.stride = stride

#         all_rms = [item["rms"].item() for item in cached_list]
#         rms_tensor = torch.tensor(all_rms, dtype=torch.float32).unsqueeze(1)

#         X, Y = [], []
#         N = rms_tensor.size()
#         for i in range(0, N - window_size - output_len + 1, stride):
#             x_seq = rms_tensor[i : i + window_size]
#             y_seq = rms_tensor[i + window_size : i + window_size + output_len]
#             X.append(x_seq)
#             Y.append(y_seq)
#         self.X = torch.stack(X)
#         self.Y = torch.stack(Y)

#     def __len__(self):
#         return  self.X.size(0)
    
#     def __getitem__(self, index):
#         return self.X[index], self.Y[index]

class RMSWindowDataset(Dataset):
    def __init__(self, cached_list, window_size=5, output_len=3, stride=1):
        """
        cached_list: torch.load()로 불러온 리스트
                     각 항목에 "rms": Tensor(scalar) 필드가 있어야 함
        window_size: 과거 RMS값을 몇 개로 묶어서 입력으로 삼을지
        stride: 슬라이딩 윈도우 이동 간격
        """
        self.window_size = window_size
        self.output_len = output_len
        self.stride = stride

        all_rms = torch.cat([item["rms"] for item in cached_list]).numpy()
        rms_arr = np.array(all_rms, dtype=np.float32)

        X_windows = []
        Y_labels = []
        N = len(rms_arr)
        for i in range(0, N - window_size - output_len + 1, stride):
            x_win = rms_arr[i : i + window_size]
            Y_lbs = rms_arr[i + window_size : i + window_size + output_len]
            X_windows.append(x_win)
            Y_labels.append(Y_lbs)

        self.X = torch.tensor(np.stack(X_windows), dtype=torch.float32)
        # self.Y = torch.tensor(Y_labels, dtype=torch.float32).unsqueeze(1) # (N, 1)
        self.Y = torch.tensor(Y_labels, dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
# class UnetEncoder_vae(nn.Module):
    # def __init__(self, window_size, in_channels, embedding_dim,
    #              num_embeddings, num_residual_layers, num_residual_hiddens,
    #              gru_hidden_dim=64, output_dim=1, pred_len=3):
    #     super().__init__()

    #     base_unet_vae = UnetEncoderVAE(
    #         in_length=window_size,
    #         in_channels=in_channels,
    #         num_residual_layers=num_residual_layers,
    #         num_residual_hiddens=num_residual_hiddens,
    #         num_embeddings=num_embeddings,
    #         embedding_dim=embedding_dim
    #     )
    #     self.encoder = base_unet_vae.encoder
    #     self.vae = VAE(embedding_dim)

    #     # GRU predictor
    #     self.predictor = GRUPredictor(
    #         input_dim=embedding_dim,
    #         hidden_dim=gru_hidden_dim,
    #         output_dim=output_dim,
    #         num_layers=1,
    #         pred_len=pred_len
    #     )

    # def forward(self, x):
    #     """
    #     x: [B, window_size]  # 1차원 RMS 시퀀스 윈도우
    #     1) UNet encoder expects [B, in_channels, in_length]
    #        → reshape x → x_in
    #     2) encoder(x_in) → z_e: [B, embedding_dim, num_embeddings]
    #     3) vae(z_e)      → z_v: [B, embedding_dim, num_embeddings]
    #     4) z_v permute   → [B, num_embeddings, embedding_dim]
    #     5) predictor(...)→ y_pred
    #     """
    #     B = x.size(0)

    #     # RMS 윈도우를 1채널 1D tensor로 변환
    #     x_in = x.unsqueeze(1) # [B, 1, window_size]

    #     # Encoder output: [B, embedding_dim, num_embeddings]
    #     z_e = self.encoder(x_in)

    #     # VAE bottleneck: z_e -> z_v, kl
    #     z_v, kl = self.vae(z_e)
        
    #     z_seq = z_v.permute(0, 2, 1)

    #     # GRU predictor -> 다음 RMS 값 예측
    #     y_pred = self.predictor(z_seq)
    #     return y_pred.squeeze(1), kl

def train_unet_gru_rms(
    train_path, val_path, test_path,
    window_size=5, in_channels=1, embedding_dim=32,
    num_embeddings=2, num_residual_layers=2, num_residual_hiddens=4,
    gru_hidden_dim=64, pred_len=1, num_epochs=30, batch_size=64,
    lr=1e-3, stride=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = load_rms_cache(train_path)
    val_raw   = load_rms_cache(val_path)
    test_raw  = load_rms_cache(test_path)

    train_ds = RMSWindowDataset(train_raw, window_size, pred_len, stride)
    val_ds   = RMSWindowDataset(val_raw,   window_size, pred_len, stride)
    test_ds  = RMSWindowDataset(test_raw,  window_size, pred_len, stride)

    print("Train dataset size:", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = UnetRMSPredictor(
        in_length=window_size,
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        gru_hidden_dim=gru_hidden_dim,
        pred_len=pred_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # [B, window_size]
            yb = yb.to(device)  # [B, 1]

            optimizer.zero_grad()
            y_pred = model(xb)           # y_pred: [B, pred_len, 1]
            y_pred = y_pred.squeeze(-1)
            # print(f"y_pred.shape: {y_pred.shape}, yb.shape: {yb.shape}")
            loss = criterion(y_pred, yb)
            # loss = loss_pred + 0.01 * kl      # KL 용도로 가중치 0.01을 곱함
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                y_pred = y_pred.squeeze(-1)
                val_loss += criterion(y_pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Epoch {epoch:02d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_gru_rms.pth")

    # --- Test Evaluation ---
    model.load_state_dict(torch.load("best_unet_gru_rms.pth"))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            y_pred = y_pred.squeeze(-1)
            preds.append(y_pred.cpu())
            targets.append(yb.cpu())

    preds = torch.cat(preds).squeeze().numpy().reshape(-1, pred_len)
    targets = torch.cat(targets).squeeze().numpy().reshape(-1, pred_len)

    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print(f"Test MAE: {mae:.6f} | Test RMSE: {rmse:.6f}")
    print("\n[예측값 vs 실제값 (상위 10개 샘플)]")
    for i in range(min(10, len(preds))):
        print(f"샘플 {i+1}:")
        print(f"  예측값: {np.round(preds[i], 4)}")
        print(f"  실제값: {np.round(targets[i], 4)}")

    return model, mae, rmse

if __name__ == "__main__":
    train_pt = "cachedRMS/Train_Set_RMS.pt"
    val_pt   = "cachedRMS/Validation_Set_RMS.pt"
    test_pt  = "cachedRMS/Test_Set_RMS.pt"

    model, mae, rmse = train_unet_gru_rms(
        train_path=train_pt,
        val_path=val_pt,
        test_path=test_pt,
        window_size=10,
        in_channels=1,
        embedding_dim=32,
        num_embeddings=2,
        num_residual_layers=2,
        num_residual_hiddens=4,
        gru_hidden_dim=32,
        pred_len=10,
        num_epochs=25,
        batch_size=128,
        lr=1e-3,
        stride=100
    )
## Seq2Seq 방식으로 변환 필요!