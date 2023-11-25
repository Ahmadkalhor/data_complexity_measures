import torch

class ARH_SeparationIndex:
    def __init__(self, data, label, normalize=False , batch_size=1000):
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
        self.batch_size = batch_size
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

    def high_order_si(self, order):
        high_si_data = torch.zeros(self.n_data, device=self.device)
        for i in range(0, self.n_data, self.batch_size):
            batch = self.data[i:i + self.batch_size]
            dis_matrix = torch.cdist(batch, self.data, p=2)
            dis_matrix.fill_diagonal_(self.big_number)
            _, arg_sort = torch.sort(dis_matrix, 1)
            sorted_labels = self.label[arg_sort[:, :order]]
            matches = (sorted_labels == self.label[i:i + self.batch_size].unsqueeze(1)).all(dim=2)
            high_si_data[i:i + self.batch_size] = matches.all(dim=1).float()
        high_si = high_si_data.mean()
        return high_si

    def high_order_si_data(self, order):
        high_si_data = torch.zeros(self.n_data, device=self.device)
        for i in range(0, self.n_data, self.batch_size):
            batch = self.data[i:i + self.batch_size]
            dis_matrix = torch.cdist(batch, self.data, p=2)
            dis_matrix.fill_diagonal_(self.big_number)
            _, arg_sort = torch.sort(dis_matrix, 1)
            sorted_labels = self.label[arg_sort[:, :order]]
            matches = (sorted_labels == self.label[i:i + self.batch_size].unsqueeze(1)).all(dim=2)
            high_si_data[i:i + self.batch_size] = matches.all(dim=1).float()
        return high_si_data
