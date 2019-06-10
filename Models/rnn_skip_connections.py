import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.utils.name_dataset import NameDataset
from Data.utils.names_dataloader import NameDataLoader

class RNN_SKIP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_SKIP, self).__init__()
        self._hidden_state_size = hidden_size
        self._i2h = nn.Linear(input_size,hidden_size)
        self._h2h_1 = nn.Linear(hidden_size,hidden_size)
        self._h2h_2 = nn.Linear(hidden_size,hidden_size)
        self._h2o = nn.Linear(hidden_size,output_size)
        self._softmax = nn.LogSoftmax(dim = 1)
        self._hidden_state = None
        self._skip_hidden_state = None

    def forward(self, batch):
        batch_size = batch.size()[1]
        seq_length = batch.size()[0]
        self._hidden_state = self.init_hidden_state(batch_size)
        for idx in range(seq_length):
            input = batch[idx,:,:]
            if idx%2 == 0:
                self._skip_hidden_state = self._hidden_state
                self._hidden_state = torch.relu(torch.add(self._i2h(input),self._h2h_1(self._hidden_state)))
                if idx == (seq_length - 1):
                    out = self._softmax(self._h2o(self._hidden_state))
                    return out
            else:
                self._hidden_state = torch.add(torch.relu(torch.add(self._i2h(input), self._h2h_2(self._hidden_state))),self._skip_hidden_state)
                if idx == (seq_length - 1):
                    out = self._softmax(self._h2o(self._hidden_state))
                    return out


    def init_hidden_state(self,batch_size):
        return torch.zeros([batch_size,self._hidden_state_size], dtype = torch.float32)
