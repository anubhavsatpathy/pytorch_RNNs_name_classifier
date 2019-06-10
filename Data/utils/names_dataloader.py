import torch
from Data.utils.name_dataset import NameDataset
from Data.utils.text_utils import labels_to_one_hot
import numpy as np

class NameDataLoader:

    def __init__(self, dataset = NameDataset(), batch_size = 30,mode = 'TRAIN'):

        assert isinstance(dataset,NameDataset),'Please make sure to instantiate with a proper <NameDataset> instance'
        self._dataset = dataset
        self._batch_size = batch_size
        self._char_id_map = dataset.get_char_id_map()
        self._mode = 'TRAIN'
        self._num_batches_per_epoch = int(len(dataset)/batch_size) + 1
        self._i = 0
        self._max_seq_length = dataset.get_max_seq_length()
        self._batch_tensor = None

    def toggle_mode(self,mode):
        assert mode in ['TRAIN','TEST']
        self._dataset.toggle_mode(mode)
        self._mode = mode
        self._num_batches_per_epoch = int(len(self._dataset) / self._batch_size)

    def get_batch_size(self):
        return self._batch_size

    def char_to_one_hot(self,ch):

        #print(ch)
        cht = torch.zeros(size=[1,self._dataset.get_num_chars()],dtype=torch.float32)
        cht[0,self._char_id_map[ch]] = 1.0
        return cht

    def __iter__(self):

        return self

    def __next__(self):
        #print(self._i)
        while self._i <= self._num_batches_per_epoch:
            batch_tensor = None
            labels = []
            for k in range(self._batch_size):
                name,cls = self._dataset[k]
                labels.append(cls)
                name_tensor_list = []
                for ch in name:
                    ch_one_hot = self.char_to_one_hot(ch)
                    name_tensor_list.append(ch_one_hot)
                while len(name_tensor_list) != self._max_seq_length:
                    name_tensor_list.append(self.char_to_one_hot('EOW'))
                stacked_name_tensor = torch.stack(name_tensor_list).squeeze().resize_([1,self._max_seq_length,self._dataset.get_num_chars()])
                #print('Stacked Name Tensor : {}'.format(stacked_name_tensor.size()))
                if batch_tensor is None:
                    batch_tensor = stacked_name_tensor
                    batch_tensor.resize_([1,self._max_seq_length,self._dataset.get_num_chars()])
                    #print(batch_tensor.size())
                else:
                    batch_tensor = torch.cat([batch_tensor,stacked_name_tensor],dim=0)
                    #print(batch_tensor.size())
            self._i += 1
            p = torch.randperm(self._batch_size)
            labels = torch.tensor(labels)
            labels = labels[p]
            batch_tensor = batch_tensor[p]
            return np.transpose(batch_tensor,(1,0,2)),labels_to_one_hot(labels,self._dataset.get_num_classes())
        self._i = 0
        self._dataset.flush_i()
        raise StopIteration



if __name__ == '__main__':

    names = NameDataset()
    loader = NameDataLoader(dataset=names)
    it = iter(loader)
    batch,labels = next(it)
    print(batch.size()[1])
    bat = batch.numpy()
    for k in range(30):
        name = ''
        for vec in bat[:,k]:
            label = labels[k]
            label = torch.argmax(label)
            label = label.item()
            idx = np.argmax(vec)
            ch = names.get_id_char_map()[idx]
            if ch != 'EOW':
                name += ch
        print(name + ' : '  + names.get_classes()[label])

    hidden = torch.zeros([30,30])
    input = batch[1]
    print(input.size())
    combined = torch.cat((batch[1,:,:],hidden), dim = 1)
    print(combined.size())







