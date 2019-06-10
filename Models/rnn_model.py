import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.utils.name_dataset import NameDataset
from Data.utils.names_dataloader import NameDataLoader

class RNN(nn.Module):
    def __init__(self, hidden_size,input_size,output_size):
        super(RNN, self).__init__()
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layer = None
        self._i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self._i2o = nn.Linear(input_size + hidden_size, output_size)
        self._softmax = nn.LogSoftmax(dim = 1)

    def forward(self, batch):
        batch_size = batch.size()[1]
        self._hidden_layer = self.init_hidden_state(batch_size)
        seq_length = batch.size()[0]
        for item in range(seq_length):
            input = batch[item,:,:]
            combined = torch.cat((input,self._hidden_layer),dim=1)
            self._hidden_layer = F.tanh(self._i2h(combined))
            output = self._i2o(combined)
            output = self._softmax(output)
        return output,self._hidden_layer

    def init_hidden_state(self,batch_size):
        return torch.zeros([batch_size,self._hidden_size],dtype=torch.float32)


if __name__ == '__main__':

    names = NameDataset()
    name_loader = NameDataLoader(dataset=names)

    name_iter = iter(name_loader)
    batch,labels = next(name_iter)

    input = batch[1,:,:]
    hidden = torch.zeros([30,60])
    output_size = names.get_num_classes()

    net = RNN(60,90,output_size)

    out,hid = net(batch)

    print(hid.size())
    print(out.size())
