import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from .basenet import *
from torch import nn, Tensor
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2400):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEncoding_query(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class model_crossFeat(torch.nn.Module):
    def __init__(self, args):
        super(model_crossFeat, self).__init__()

        num_classes = 1
        hidden_dim = args.hidden_dim
        nheads = args.num_heads
        num_encoder_layers = args.num_encoder_layers
        num_decoder_layers = args.num_decoder_layers
        self.use_rgb = False

        # activation function
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()
        self.dropout_cnn = args.dropout_cnn
        self.sine_flag = args.sine_flag
        self.th = args.th
        self.mask = args.mask
        self.use_titan = args.titan

        if self.use_titan:
            self.num_features = 3
        else:
            self.num_features = 5

        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward=args.feedforward_dim, dropout=args.dropout_transformer)
        # self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
        #                                   dim_feedforward=args.feedforward_dim, dropout=args.dropout_transformer)

        # create a transformer decoder
        self.decoder = nn.TransformerDecoderLayer(hidden_dim, 1, dim_feedforward=args.feedforward_dim, dropout=args.dropout_transformer)
        self.query_parameter = nn.Parameter(torch.rand(1, 1, hidden_dim))

        in_channels = 10
        self.H_transformer = 30
        self.W_transformer = 80
        if self.sine_flag:
            self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0.1, max_len= self.H_transformer * self.W_transformer)
        else:
            self.row_embed = nn.Parameter(torch.rand(self.H_transformer, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(self.W_transformer, hidden_dim // 2))
        self.positional_encoding_query = PositionalEncoding_query(hidden_dim, dropout=0.1, max_len=self.num_features)

        if self.use_rgb:
            res18 = ResnetBlocks(torchvision.models.resnet18(pretrained=True))
            self.resnet = res18.build_backbone(use_pool=False, use_last_block=False, pifpaf_bn=False)
            self.layer_cnn_3 = res18.block5()
            self.layer_fc_roi = nn.Sequential(
                nn.Linear(in_features=512*4*4, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        else:
            self.layer_cnn_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout_cnn),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.layer_cnn_2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout_cnn)
            )

            self.layer_cnn_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout_cnn)
            )


            dim_roi = 32
            self.layer_fc_roi = nn.Sequential(
                nn.Linear(in_features=dim_roi*4*4, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        if self.use_rgb:
            self.conv = nn.Sequential(nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(32, hidden_dim, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))

        #motion
        self.fc_pv = nn.Sequential(nn.Linear(8, hidden_dim), self.relu)
        self.rnn_pv = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        #behavior
        self.fc_beh = nn.Sequential(nn.Linear(4, hidden_dim), self.relu)
        self.rnn_beh = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        #speed
        self.fc_speed = nn.Sequential(nn.Linear(1, hidden_dim), self.relu)
        self.rnn_speed = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        #scene attribute
        self.fc_scene_attributes = nn.Sequential(nn.Linear(6, hidden_dim), self.relu)

        self.fc_last = nn.Sequential(nn.Linear(hidden_dim, num_classes))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        #self.linear_class = nn.Linear(hidden_dim, num_classes)

    def speed_feature(self, pv, x_len):
        pv_feature = self.fc_speed(pv.unsqueeze(2).float())
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(pv_feature, x_len, batch_first=True, enforce_sorted=False)
        packed_RNN_out_1, _ = self.rnn_speed(packed_x1_RNN)
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        #pv_feat = RNN_out_1[:, -1].unsqueeze(0)
        pv_feat = RNN_out_1[torch.arange(x_len.shape[0]), x_len-1, :].unsqueeze(0)
        return pv_feat

    def pose_feature(self, pv, x_len):
        pv_feature = self.fc_pose(pv.float())
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(pv_feature, x_len, batch_first=True, enforce_sorted=False)
        packed_RNN_out_1, _ = self.rnn_pose(packed_x1_RNN)
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        #pv_feat = RNN_out_1[:, -1].unsqueeze(0)
        pv_feat = RNN_out_1[torch.arange(x_len.shape[0]), x_len-1, :].unsqueeze(0)
        return pv_feat

    def motion_feature(self, pv, x_len):
        pv_feature = self.fc_pv(pv)
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(pv_feature, x_len, batch_first=True, enforce_sorted=False)
        packed_RNN_out_1, _ = self.rnn_pv(packed_x1_RNN)
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        #pv_feat = RNN_out_1[:, -1].unsqueeze(0)
        pv_feat = RNN_out_1[torch.arange(x_len.shape[0]), x_len-1, :].unsqueeze(0)
        return pv_feat

    def behavior_feature(self, behavior, x_len):
        beh_feature = self.fc_beh(behavior)
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(beh_feature, x_len, batch_first=True, enforce_sorted=False)
        packed_RNN_out_1, _ = self.rnn_beh(packed_x1_RNN)
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        #beh_feat = RNN_out_1[:, -1].unsqueeze(0)
        beh_feat = RNN_out_1[torch.arange(x_len.shape[0]), x_len-1, :].unsqueeze(0)
        return beh_feat

    def forward(self, imgs, bboxes, pv, behavior, scene, pose=None, speed=None, x_len=None, training_flag=False, freeze_feature=None):
        bs = imgs.shape[0]
        if self.use_rgb:
            x_2 = self.resnet(imgs)[:,:,:-1,:-1]
        else:
            x_1 = self.layer_cnn_1(imgs)
            x_2 = self.layer_cnn_2(x_1)
        fa = RoIAlign(output_size=(7, 7), spatial_scale=1 / 8, sampling_ratio=2, aligned=True)
        ya = fa(x_2, bboxes)
        x_3 = self.layer_cnn_3(ya)

        if self.use_rgb:
            x_3 = x_3.reshape(-1, 512 * 4 * 4)
        else:
            x_3 = x_3.reshape(-1, 32 * 4 * 4)
        feat_bbox = self.layer_fc_roi(x_3)

        #input feature for transformer
        feature_maps_input = self.conv(x_2)

        if self.sine_flag:
            inputs = self.positional_encoding(feature_maps_input.flatten(2).permute(2, 0, 1))
        else:
            H, W = feature_maps_input.shape[-2:]
            pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1).flatten(0, 1).unsqueeze(1)
            pos = pos.repeat(1,bs,1)
            inputs = pos + feature_maps_input.flatten(2).permute(2, 0, 1)

        #motion and behavior feature
        bbox_feature = feat_bbox.unsqueeze(0)
        motion_feature = self.motion_feature(pv, x_len)
        speed_feature = self.speed_feature(speed, x_len)

        if self.use_titan:
            inputs_query = torch.cat((bbox_feature, motion_feature, speed_feature), dim=0)
        else:
            behavior_feature = self.behavior_feature(behavior, x_len)
            scene_feature = self.fc_scene_attributes(scene).unsqueeze(0)
            inputs_query = torch.cat((bbox_feature, motion_feature, behavior_feature, scene_feature, speed_feature), dim=0)
        inputs_query = self.positional_encoding_query(inputs_query)

        h = self.transformer(inputs, inputs_query)

        #classification layer
        output_decoder = self.decoder(self.query_parameter.expand(-1, bs, -1), h)

        y = self.act(self.mlp_head(output_decoder)).squeeze()

        return y


