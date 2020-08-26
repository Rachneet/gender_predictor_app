import torch.nn as nn
import torch
import numpy as np
from nltk.corpus import stopwords
import string
import sys

# -----------------------------------------------ML model-------------------------------------------------------------
# char level cnn model
class CharCNN(nn.Module):

    def __init__(self, n_classes, input_dim=69, max_seq_length=1014, filters=256, kernel_sizes=[7, 7, 3, 3, 3, 3],
                 pool_size=3,
                 n_fc_neurons=1024):

        super(CharCNN, self).__init__()

        self.filters = filters
        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.pool_size = pool_size

        # pooling in layer 1,2,6 ; pool =3
        # layers 7,8 and 9 are fully connected
        # 2 dropout modules between 3 fully connected layers
        # dropout - prevents overfitting in neural networks ; drops neurons with a certain probability
        # (from 2014 paper Dropout by Srivastava et al.); only during training

        # layer 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, filters, kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # layer 3,4,5
        self.conv3 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[2]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[3]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[4]),
            nn.ReLU()
        )

        # layer 6
        self.conv6 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_sizes[5]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        dimension = int((max_seq_length - 96) / 27 * filters)

        # layer 7
        self.fc1 = nn.Sequential(
            nn.Linear(dimension, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # layer 8
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # layer 9
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

        if filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):

        input = input.transpose(1, 2)
        # print(input.size())
        output = self.conv1(input)
        # print(output.size())
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.sigmoid(output)

        return output


def predict_gender(text):
    res = ""
    words = stopwords.words("english")
    table = str.maketrans('', '', string.punctuation)
    # text = "This thing is not good at all. Do not buy it."
    text = text[0]
    cleaned_text = " ".join([i.translate(table) for i in text.split() if i.isalpha() if i not in words]).lower()
    max_length = 1014
    vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
    identity_mat = np.identity(len(vocabulary))

    data = np.array([identity_mat[vocabulary.index(i)] for i in list(cleaned_text) if i in vocabulary],
                    dtype=np.float32)

    if len(data) > max_length:
        data = data[:max_length]
    elif 0 < len(data) < max_length:
        data = np.concatenate(
            (data, np.zeros((max_length - len(data), len(vocabulary)), dtype=np.float32)))
    elif len(data) == 0:
        data = np.zeros((max_length, len(vocabulary)), dtype=np.float32)

    model = torch.load("model_amz_ccnn").cpu()
    model.eval()

    data = torch.FloatTensor(data).cpu()
    data = data.unsqueeze(dim=0)
    # print(data.shape)
    prediction = model(data)
    # print(prediction)
    prediction = torch.argmax(prediction)
    if prediction == 0:
        res = "Male"
    else:
        res = "Female"

    return res


if __name__=="__main__":
    obj = CharCNN(n_classes=2)
