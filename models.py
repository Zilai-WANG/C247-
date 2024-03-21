import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers
import tensorflow.keras.backend as K
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# fix CUDNN_STATUS_INTERNAL_ERROR
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
DROPOUT = 0.4


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)
################################################################
''' GRU'''




class ResNetGRU(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained=True):
        super(ResNetGRU, self).__init__()

        # Load a pretrained ResNet and remove the fully connected layer
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Identity()

        # Assuming resnet18 outputs 512 features
        resnet_features = 512

        # GRU Layer
        self.gru = nn.GRU(resnet_features, 128, 3, batch_first=True, dropout=dropout_rate)
        # self.gru = nn.GRU(resnet_features, 64, 3, batch_first=True, dropout=dropout_rate)
        # # Fully Connected Layer
        # self.fc = nn.Sequential(
        #     nn.Linear(64, 54),
        #     # nn.BatchNorm1d(54, eps=1e-05, momentum=0.2, affine=True),
        #     nn.LayerNorm(54),  # Apply LayerNorm
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(54, 32),
        #     nn.LayerNorm(32),  # Apply LayerNorm
        #     # nn.BatchNorm1d(32, eps=1e-05, momentum=0.2, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 4)
        # )
        self.fc = nn.Sequential(
          nn.Linear(128, 64),
          nn.LayerNorm(64),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Dropout(p=dropout_rate),
          nn.Linear(64, 32),
          nn.LayerNorm(32),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Dropout(p=dropout_rate),
          nn.Linear(32, 16),
          nn.LayerNorm(16),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Linear(16, 4)
        )

    def forward(self, x):
        # print("asdadsda\n\n")
        # print(x.size())
        x = x.repeat(1, 3, 1, 1)  # Repeats the channel dimension 3 times
        # print(x.size())
    # Directly use ResNet on the input x with shape (N, C, H, W)
        c_out = self.resnet(x)  # Output shape: (N, feature_size)

        # Introduce a sequence length dimension for GRU processing
        # After ResNet, reshape c_out to add a sequence length of 1: (N, 1, feature_size)
        r_out = c_out.unsqueeze(1)

        # GRU processing
        out, _ = self.gru(r_out)

        # Fully Connected Layer processing
        # Take the output of the last (and only) time step
        out = self.fc(out[:, -1, :])  # This simplifies to maintaining the shape (N, feature_size)

        return out



################################################################
'''
LSTM
'''
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        #print("Input to EncoderLSTM shape:", x.shape) #Input to EncoderLSTM shape: torch.Size([32, 1, 22, 1000])
        x = x.squeeze(1)  # This removes the second dimension
        x = x.permute(0, 2, 1)  # Correctly reorder dimensions to LSTM's expected input format of (batch, seq, feature) = [32,1000,22]
        outputs, (hidden, cell) = self.lstm(x) 
        #print("EncoderLSTM outputs shape:", outputs.shape) #EncoderLSTM outputs shape: torch.Size([32, 1000, hidden])
        #print("EncoderLSTM hidden state shape:", hidden.shape)#EncoderLSTM hidden state shape: torch.Size([num_layer, 32, hidden])
        #print("EncoderLSTM cell state shape:", cell.shape)#EncoderLSTM cell state shape: torch.Size([num_layer, 32, hidden])
        return outputs, (hidden, cell)
        
        
class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(DecoderLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Ensure the LSTM layer's input size is set to hidden_size + output_size
        self.lstm = nn.LSTM(hidden_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        #print("Input to DecoderLSTMWithAttention shape:", input.shape) #Input to DecoderLSTMWithAttention shape: torch.Size([32, 1, hidden])
        attn_weights = self.attention(hidden[-1], encoder_outputs)#hidden state shape: torch.Size([num_layer, 32, hidden])
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)#attention weights size is:  torch.Size([32, 1000]) and #EncoderLSTM outputs shape: torch.Size([32, 1000, hidden])
        rnn_input = torch.cat((input, context), -1)
        #print("DecoderLSTMWithAttention concatenated input shape:", rnn_input.shape)#DecoderLSTMWithAttention concatenated input shape: torch.Size([32, 1, 2*hidden])
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        #print("DecoderLSTMWithAttention output shape:", output.shape) #DecoderLSTMWithAttention output shape: torch.Size([32, 1, hidden])
        output = self.fc(output.squeeze(1))
        #print("After FC layer output shape:", output.shape) #After FC layer output shape: torch.Size([32, 4])
        return output, hidden, cell, attn_weights
        
        
class Seq2SeqForClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(Seq2SeqForClassification, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderLSTMWithAttention(output_size, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg=None):
        #print("Input to Seq2Seq model shape:", src.shape) #Input to Seq2Seq model shape: torch.Size([32, 1, 22, 1000])
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = torch.zeros(src.size(0), 1, self.decoder.hidden_size).to(src.device)
        
        output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs) # output shape: torch.Size([32, 4])
        #output = self.fc(hidden[-1].squeeze(0)) #hidden [num_layer,32,hidden]
        output = self.fc(hidden[-1]) #hidden [num_layer,32,hidden]
        #print("Seq2Seq final output shape:", output.shape) #Seq2Seq final output shape: torch.Size([32, 4])
        return output


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # Ensure hidden is from the last layer, shape: [batch_size, hidden_size]
        # if hidden.dim() == 3:  # multi-layer scenario
        #     hidden = hidden[-1]  # Take hidden state of the last layer
        
        #print('attention hidden size is: ',hidden.shape) #attention hidden size is:  torch.Size([32, hidden])
        hidden = hidden.unsqueeze(2)#[32, 16, 1] #encoder_outputs:torch.Size([32, 1000, 16])
        attn_weights = torch.bmm(encoder_outputs, hidden).squeeze(2)
        #print('attention weight size before softmax is: ',attn_weights.shape) #hidden size is:  torch.Size([32, hidden])
        #attention weight size before softmax is:  torch.Size([32, 1000])
        attn_weights = F.softmax(attn_weights, dim=1)
        #print('attention weight size after softmax is: ',attn_weights.shape) #hiden size is:  torch.Size([32, hidden])
        #attention weight size before softmax is:  torch.Size([32, 1000])
        return attn_weights



# class LSTM(nn.Module):
#     def __init__(self, dropout_rate=0.3):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(22, 64, 2, batch_first=True, dropout=dropout_rate)
#         self.fc = nn.Sequential(
            
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
#             nn.ReLU(inplace = True),
#             nn.Linear(32, 4)
#         )
    
#     def forward(self, x):
#         N, C, H, W = x.size()
#         print(x.size())
#         x = x.view(N, H, W).permute(0, 2, 1)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
########################################################








