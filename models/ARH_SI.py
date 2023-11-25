import torch

class ARH_SeparationIndex:
    def __init__(self, data, label, normalize=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data, label = data.to(self.device), label.to(self.device)

        self.normalize = normalize
        if normalize:
            mean_data = data.mean(0, keepdim=True)
            std_data = data.std(0, keepdim=True) + 1e-10
            self.data = (data - mean_data) / std_data
        else:
            self.data = data

        self.label_min = label.min().item()
        self.label = label - self.label_min
        self.big_number = 1e10
        self.dis_matrix = torch.cdist(self.data, self.data, p=2)
        self.dis_matrix.fill_diagonal_(self.big_number)
        self.n_class = self.label.max().item() + 1
        self.n_data = self.data.shape[0]
        self.n_feature = self.data.shape[1]

    def si(self):
        values, indices = torch.min(self.dis_matrix, 1)
        si_data = (self.label[indices] == self.label)
        si = si_data.sum().float() / self.n_data
        return si

    def si_data(self):
        values, indices = torch.min(self.dis_matrix, 1)
        si_data = (self.label[indices] == self.label).float()
        return si_data
