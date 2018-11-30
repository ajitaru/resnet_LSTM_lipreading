import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self, num_frames):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss()
        self.num_frames = num_frames

    def forward(self, input, target):
        loss = 0.0
        transposed = input.transpose(0, 1).contiguous()
        for i in range(0, self.num_frames):
            loss += self.criterion(transposed[i], target)

        return loss

def _validate(modelOutput, labels):
    # modelOutput               # num_batch x 29 x 500
    # labels                    # num_batch x 1
    averageEnergies = torch.sum(modelOutput.data, 1)            # num_batch x 500
    maxvalues, maxindices = torch.max(averageEnergies, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1

    return count, maxindices

class LSTMBackend(nn.Module):
    def __init__(self, options):
        super(LSTMBackend, self).__init__()
        self.Module1 = nn.LSTM(input_size=options["model"]["inputdim"], hidden_size=options["model"]["hiddendim"],
                                num_layers=options["model"]["numlstms"], batch_first=True, bidirectional=True)

        self.fc = nn.Linear(options["model"]["hiddendim"] * 2, options["model"]["numclasses"])
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = NLLSequenceLoss(options["general"]["num_frames"])
        self.validator = _validate

    def forward(self, input):
        temporalDim = 1
        lstmOutput, _ = self.Module1(input)
        output = self.fc(lstmOutput)
        output = self.softmax(output)
        return output
