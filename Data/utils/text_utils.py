import torch

def get_num_lines(filepath):
    num_lines = 0;
    with open(filepath) as file:
        for line in file:
            num_lines += 1

    file.close()
    return num_lines

def labels_to_one_hot(labels, num_classes):
    one_hot = None
    for label in labels:
        x = torch.zeros([1,num_classes], dtype = torch.int64)
        x[0,label] = 1.0
        if one_hot is None:
            one_hot = x
            #print(one_hot)
        else:
            one_hot = torch.cat((one_hot,x),dim = 0)
    return one_hot

def topk(labels, output,k = 5):
    assert labels.size()[1] > k, 'Number of classes must be greater than k'
    top_k = torch.topk(output,k=k)
    top_k_indices = top_k[1]
    correct = 0
    for i in range(k):
        correct += (top_k_indices[:,i] == labels.argmax(dim = 1)).sum()
    return correct


if __name__ == "__main__":

    print(get_num_lines(filepath = '/Users/satpathya/Dev/RNN_name_classifier/Data/names/Arabic.txt'))
    labels = [1,2,3,4,5,6,5,4,3,2]
    logits = labels_to_one_hot(labels,10)
    print(logits)