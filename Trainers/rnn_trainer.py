import torch
import torch.nn as nn
import torch.optim as optim
from Data.utils.name_dataset import NameDataset
from Data.utils.names_dataloader import NameDataLoader
from Data.utils.text_utils import topk
from Models.rnn_model import RNN
from Models.lstm_model import LSTM
from Models.rnn_skip_connections import RNN_SKIP

def train(dataset = NameDataset(), dataloader = NameDataLoader(), epochs = 100,net = RNN(hidden_size=128,input_size=90,output_size=18), input_size =  90, hidden_size = 60):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.05,weight_decay=0.0)
    batch_num = 0
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for batch,labels in dataloader:
            optimizer.zero_grad()
            batch_num += 1
            out = net(batch)
            loss = criterion(out,labels.argmax(dim=1))
            loss.backward()
            #nn.utils.clip_grad_norm_(net.parameters(),0.25)
            optimizer.step()
            running_loss += loss.item()
            #print('Batch Number : {}'.format(batch_num))
            #print('Loss = {} ---- Running loss = {}'.format(loss.item(),running_loss))
            #print('Gradients : ')
            #for name, p in net.named_parameters():
             #   print(name)
              #  print(p.grad)
            correct += topk(labels,out)
            total += dataloader.get_batch_size()
            if batch_num == 1:
                print('[Epoch : {} -- Batch : {} -- Running Loss = {} -- Correct ={} -- Total = {}]'.format(epoch + 1,
                                                                                                            batch_num,
                                                                                                            running_loss,
                                                                                                            correct,
                                                                                                            total))

        print('[Epoch : {} -- Batch : {} -- Running Loss = {} -- Correct ={} -- Total = {}]'.format(epoch + 1, batch_num,
                                                                                                    running_loss,
                                                                                                    correct, total))
        print(labels.argmax(dim=1) == out.argmax(dim=1))


if __name__ == '__main__':
    names = NameDataset()
    loader = NameDataLoader(dataset=names, batch_size=100)
    net = LSTM(cell_state_size=100, hidden_state_size=100, input_size=90, output_size=18)
    net_skip = RNN_SKIP(input_size=90,hidden_size=120,output_size=18)
    train(dataset=names,dataloader=loader,net=net_skip)



