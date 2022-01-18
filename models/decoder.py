import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.gcnn import GCNN
from utils import gcn_tools


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu",
                 dataname='ETTh1'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # 增加GCN attention 部分的内容

        self.dec_gcnn_layer = GCNN(data=dataname,
                                   in_channels=1,
                                   out_channels=d_model)
        # self.dec_conv1 = nn.Conv1d(in_channels=d_model * adj_matrix.shape[0],
        #                            out_channels=d_model,
        #                            kernel_size=3, padding=1, padding_mode='circular')
        self.dec_conv1 = nn.Conv1d(in_channels=d_model,
                                   out_channels=self.dec_gcnn_layer.get_adj_matrix_shape()[0],
                                   kernel_size=1)

        self.dec_conv2 = nn.Conv1d(in_channels=d_model * self.dec_gcnn_layer.get_adj_matrix_shape()[0],
                                   out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')

        self.projection_dec = nn.Linear(d_model * 2, d_model,bias=True)


    def forward(self, x, x_dec, cross, x_mask=None, cross_mask=None):
        # x【32，72，512】，cross【32，48，512】
        # decoder 部分
        # B, L, D = 32,72,512

        # 做一次特征蒸馏
        # gcn_in = self.dec_conv1(x.permute(0, 2, 1)).transpose(1, 2)
        B, L, D = x_dec.shape
        # gcn_in shape is [32, 512, 72, 1]
        gcn_in = x_dec.transpose(1, 2).reshape(B, D, L, 1)
        # gcn_out shape is [32, 72, 7*512]
        gcn_out = self.dec_gcnn_layer(gcn_in).transpose(1, 2).reshape(B, L, -1)

        # project 得到【32，72，512】输出
        gcn_out = self.dec_conv2(gcn_out.permute(0, 2, 1)).transpose(1, 2)
        # gcn_out = self.dec_conv2(gcn_out)



        x = x + gcn_out + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + gcn_out + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # print("拼接前的维度：",x.shape)
        # x = torch.cat((x,gcn_out),dim = 2)
        # x = self.projection_dec(x)
        # print("拼接后的维度：",x.shape)

        # exit()

        # x = x + self.dropout(self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask
        # )[0])
        # x = self.norm1(x)
        #
        #
        #
        #
        # x = x + self.dropout(self.cross_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, x_dec, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, x_dec, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
