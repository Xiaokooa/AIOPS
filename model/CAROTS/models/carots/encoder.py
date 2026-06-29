import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.data.data import Data

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
from models.timesnet.modeling_timesnet import TimesBlock


class iTransformer_ENC(nn.Module):
    def __init__(self, cfg):
        super(iTransformer_ENC, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.ITRANSFORMER

        self.enc_embeddings = nn.ModuleList([
            DataEmbedding_inverted(
                self.cfg.DATA.WIN_SIZE,
                self.cfg_model.D_MODEL,
            ) for _ in range(self.cfg.DATA.N_VAR)
        ])

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.cfg_model.FACTOR,
                            attention_dropout=self.cfg_model.DROPOUT,
                            output_attention=self.cfg_model.OUTPUT_ATTENTION,
                        ),
                        self.cfg_model.D_MODEL,
                        self.cfg_model.N_HEADS,
                    ),
                    self.cfg_model.D_MODEL,
                    self.cfg_model.D_FF,
                    dropout=self.cfg_model.DROPOUT,
                    activation=self.cfg_model.ACTIVATION,
                )
                for _ in range(self.cfg_model.E_LAYERS)
            ],
            norm_layer=torch.nn.LayerNorm(self.cfg_model.D_MODEL),
        )
    
    def forward(self, x):
        enc_outs = []
        for i in range(self.cfg.DATA.N_VAR):
            enc_out = self.enc_embeddings[i](x[:, :, i:i+1], None)
            enc_outs.append(enc_out)
        enc_out = torch.cat(enc_outs, dim=1)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # average pooling
        enc_out = torch.mean(enc_out, dim=1)
        return enc_out


class TimesNet_ENC(nn.Module):
    def __init__(self, cfg):
        super(TimesNet_ENC, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.TIMESNET

        self.seq_len = self.cfg.DATA.WIN_SIZE
        self.nvar = self.cfg.DATA.N_VAR
        self.cfg_model.seq_len = self.seq_len
        self.cfg_model.enc_in = self.nvar
        self.cfg_model.c_out = self.nvar

        self.model = nn.ModuleList(
            [TimesBlock(self.cfg_model) for _ in range(self.cfg_model.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            self.cfg_model.enc_in,
            self.cfg_model.d_model,
            self.cfg_model.embed,
            self.cfg_model.freq,
            self.cfg_model.dropout,
        )
        self.layer = self.cfg_model.e_layers
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.cfg_model.d_model) for _ in range(self.cfg_model.e_layers)]
        )

    def forward(self, x):
        # embedding
        enc_out = self.enc_embedding(x, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm[i](self.model[i](enc_out))

        enc_out = torch.mean(enc_out, dim=1)
        return enc_out


class LSTM_ENC(nn.Module):
    def __init__(self, cfg):
        super(LSTM_ENC, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.LSTM

        self.lstm = nn.LSTM(
            self.cfg.DATA.N_VAR,
            self.cfg_model.HIDDEN_DIM,
            num_layers=self.cfg_model.NUM_LAYERS,
            dropout=self.cfg_model.DROPOUT,
            batch_first=True,
        )

    def forward(self, x):
        batch_size = x.size(0)
        o, (h, c) = self.lstm(x)  # o: (batch_size, time_step, hidden_dim), h: (num_layers, batch_size, hidden_dim), c: (num_layers, batch_size, hidden_dim)
        h = h.permute((1, 0, 2))  # h: (batch_size, num_layers, hidden_dim)
        enc_out = torch.reshape(h, (batch_size, -1))  # h: (batch_size, num_layers * hidden_dim)

        return enc_out



class GRU_ENC(nn.Module):
    def __init__(self, cfg):
        super(GRU_ENC, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.GRU

        self.gru = nn.GRU(
            self.cfg.DATA.N_VAR,
            self.cfg_model.HIDDEN_DIM,
            num_layers=self.cfg_model.NUM_LAYERS,
            dropout=self.cfg_model.DROPOUT,
            batch_first=True,
        )

    def forward(self, x):
        batch_size = x.size(0)
        o, h = self.gru(x)  # o: (batch_size, time_step, hidden_dim), h: (num_layers, batch_size, hidden_dim), c: (num_layers, batch_size, hidden_dim)
        h = h.permute((1, 0, 2))  # h: (batch_size, num_layers, hidden_dim)
        enc_out = torch.reshape(h, (batch_size, -1))  # h: (batch_size, num_layers * hidden_dim)

        return enc_out


class GATV2_ENC(nn.Module):
    def __init__(self, cfg):
        super(GATV2_ENC, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.GATV2
        
        win_size = self.cfg.DATA.WIN_SIZE
        in_channels = self.cfg_model.IN_CHANNELS
        out_channels = self.cfg_model.OUT_CHANNELS
        hidden_channels = self.cfg_model.HIDDEN_CHANNELS
        heads = self.cfg_model.HEADS
        dropout = self.cfg_model.DROPOUT
        
        self.temporal_embedding = nn.Linear(win_size, in_channels)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    @staticmethod
    def adjacency_matrix_to_edge_index(adj):
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    def forward(self, input, adj_mtx):  # input: (batch_size, time, num_nodes), adj_mtx: (num_nodes, num_nodes)
        bsz, time, n_nodes = input.shape
        input = self.temporal_embedding(input.transpose(1, 2))
        edge_index = self.adjacency_matrix_to_edge_index(adj_mtx)
        
        # Create a single batch with all nodes
        batch = Data(x=input.view(-1, input.size(-1)), edge_index=edge_index.repeat(1, bsz))
        
        x, edge_index = batch.x, batch.edge_index
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Reshape and average pooling
        output = torch.mean(x.view(bsz, n_nodes, -1), dim=1)

        return output
