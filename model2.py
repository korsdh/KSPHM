from unet import UnetVAE, VAE
import torch
import torch.nn as nn
import pandas as pd


class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, pred_len=10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_seq):
        """
        z_seq: [B, L, C] — latent sequence
        return: [B, pred_len, output_dim] — predicted future sequence
        """
        B, L, C = z_seq.size()
        device = z_seq.device
        
        h_t = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        for t in range(L):
            h_t = self.gru_cell(z_seq[:, t, :], h_t)

        outputs = []
        input_t = torch.zeros(B, self.input_dim, device=z_seq.device)

        for _ in range(self.pred_len):
            h_t = self.gru_cell(input_t, h_t)
            y_t = self.out_proj(h_t)
            outputs.append(y_t.unsqueeze(1))
            input_t = torch.zeros(B, self.input_dim, device=device)

        return torch.cat(outputs, dim=1)
    
class UnetEncoderVAE(nn.Module):
    def __init__(self, in_length, in_channels, embedding_dim, num_embeddings, num_residual_layers,
                 num_residual_hiddens, hidden_dim=128, output_dim=1, pred_len=1):
        super().__init__()

        self.encoder = UnetVAE(
            in_length=in_length,
            in_channels=in_channels,
            z_dim=None,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        ).encoder

        self.vae = VAE(embedding_dim)

        self.predictor = GRUPredictor(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pred_len=pred_len
        )

    def forward(self, x):
        z = self.encoder(x) # [B, C, L]
        z = z.permute(0, 2, 1) # [B, L, C]

        z_sampled, kl = self.vae(z.permute(0, 2, 1))
        z_sampled = z_sampled.permute(0, 2, 1)

        y_pred = self.predictor(z_sampled)
        return y_pred.squeeze(1), kl
    

class UnetEncoder(nn.Module):
    def __init__(self, in_length, in_channels,
                 embedding_dim, num_embeddings,
                 num_residual_layers, num_residual_hiddens):
        """
        in_length:  입력 시퀀스 길이 (window_size)
        in_channels: 입력 채널 수 (RMS 여기는 1)
        embedding_dim: 최종 feature 채널 수
        num_embeddings: 시퀀스 길이를 줄여서 남길 양 (encoder 마지막 length)
        num_residual_layers: Residual 블록 수
        num_residual_hiddens: Residual 블록 내 hidden 비율
        """
        super().__init__()
        base = UnetVAE(
            in_length=in_length, in_channels=in_channels,
            z_dim=None,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        self.encoder = base.encoder

    def forward(self, x):
        """
        x: [B, in_channels, in_length]
        
        returns:
          z_e: [B, embedding_dim, num_embeddings]
        """
        return self.encoder(x)
    

class UnetRMSPredictor(nn.Module):
    def __init__(self, in_length, in_channels,
                 embedding_dim, num_embeddings,
                 num_residual_layers, num_residual_hiddens,
                 gru_hidden_dim=64, pred_len=1):
        super().__init__()
        base = UnetVAE(
            in_length=in_length,
            in_channels=in_channels,
            z_dim=None,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        self.encoder = base.encoder

        self.predictor = GRUPredictor(
            input_dim=embedding_dim,
            hidden_dim=gru_hidden_dim,
            output_dim=1,       # RMS 는 스칼라
            num_layers=1,
            pred_len=pred_len
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.encoder(x)
        z_seq = z.permute(0, 2, 1)
        y_pred = self.predictor(z_seq)
        return y_pred.squeeze(-1)