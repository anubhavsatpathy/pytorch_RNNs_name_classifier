import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.utils.name_dataset import NameDataset
from Data.utils.names_dataloader import NameDataLoader

class LSTM(nn.Module):
    def __init__(self, cell_state_size, hidden_state_size, input_size, output_size):
        super(LSTM, self).__init__()
        assert cell_state_size == hidden_state_size
        self._cell_state = None
        self._hidden_state = None
        self._hidden_state_size = hidden_state_size
        self._cell_state_size = cell_state_size
        self.forget_layer = nn.Linear(input_size + hidden_state_size,cell_state_size)
        self._input_layer = nn.Linear(input_size + hidden_state_size,cell_state_size)
        self._candidate_layer = nn.Linear(input_size + hidden_state_size,cell_state_size)
        #self._output_layer = nn.Linear(input_size + hidden_state_size,cell_state_size)
        #self._cs2hs = nn.Linear(cell_state_size,hidden_state_size)
        self._hidden_output_layer = nn.Linear(input_size +hidden_state_size, cell_state_size)
        self._actual_output_layer = nn.Linear(hidden_state_size, output_size)
        self._softmax = nn.LogSoftmax(dim = 1)

    def forward(self, batch):
        batch_size = batch.size()[1]
        seq_length = batch.size()[0]
        self._hidden_state = self.init_hidden_state(batch_size)
        self._cell_state = self.init_cell_state(batch_size)
        for item in range(seq_length):
            input = batch[item,:,:]
            combined = torch.cat((input,self._hidden_state),dim=1)
            f_t = torch.sigmoid(self.forget_layer(combined))
            self._cell_state = torch.mul(f_t,self._cell_state)
            i_t = torch.sigmoid(self._input_layer(combined))
            candidate_t = torch.tan(self._candidate_layer(combined))
            self._cell_state = torch.add(self._cell_state,torch.mul(i_t,candidate_t))
            o_t = torch.sigmoid(self._hidden_output_layer(combined))
            #cs_hs = torch.tanh(self._cs2hs(self._cell_state))
            self._hidden_state = torch.mul(o_t,torch.tan(self._cell_state))
            if item == (seq_length -1):
                return self._softmax(self._actual_output_layer(self._hidden_state))


    def init_hidden_state(self, batch_size):
        return torch.zeros(size=[batch_size,self._hidden_state_size],dtype=torch.float32)

    def init_cell_state(self,batch_size):
        return torch.zeros(size=[batch_size,self._cell_state_size], dtype=torch.float32)

if __name__ == '__main__':

    names = NameDataset()
    name_loader = NameDataLoader(dataset=names)
    name_iterator = iter(name_loader)
    batch, labels = next(name_iterator)
    net = LSTM(cell_state_size=100,hidden_state_size=100,input_size=90,output_size=18)
    out = net(batch)
    print(out.size())
