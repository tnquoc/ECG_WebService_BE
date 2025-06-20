import torch.nn as nn


# model
class DACLSTM(nn.Module):
    def __init__(self, n_classes=10):
        super(DACLSTM, self).__init__()
        num_channel = 1
        self.channel_based_att = ChannelBasedAttentionLayer(num_channel, 1)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_channel, 16, kernel_size=5, stride=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, batch_first=True)
        self.time_based_att = TimeBasedAttentionLayer(128)
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.channel_based_att(x)
        x = self.conv_layers(x)
        x = x.transpose(2, 1)
        x, (_, _) = self.lstm(x)
        x = x.transpose(2, 1)
        x = self.time_based_att(x)
        x = self.mlp_head(x)

        return x


class ChannelBasedAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelBasedAttentionLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1)

        return x * y.expand_as(x)


class TimeBasedAttentionLayer(nn.Module):
    def __init__(self, channel):
        super(TimeBasedAttentionLayer, self).__init__()
        self.W = nn.Linear(channel, channel)
        self.U = nn.Linear(channel, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        y = self.W(x)
        y = self.tanh(y)
        y = self.U(y)
        y = self.softmax(y)
        out = x * y.expand_as(x)
        out = out.permute(0, 2, 1)

        return out


