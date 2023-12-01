import torch
from tqdm import tqdm

class Kalhor_SeparationIndex:
    def __init__(self, data, label, normalize=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.data = data.to(self.device)
        self.label = label.to(self.device)

        if normalize:
            self.normalize_data()
            print('Data has been normalized')

        self.label_min = round(torch.min(self.label).detach().item())
        self.label = (self.label - self.label_min).long()
        self.big_number = 1e10
        self.dis_matrix = torch.cdist(self.data, self.data, p=2).fill_diagonal_(self.big_number)
        self.n_class = round(torch.max(self.label).detach().item()) + 1
        self.n_data = self.data.shape[0]
        self.n_feature = self.data.shape[1]

    def normalize_data(self):
        small_number = 1e-10
        mean_data = torch.mean(self.data, dim=0)
        std_data = torch.std(self.data, dim=0) + small_number
        self.data = (self.data - mean_data) / std_data

    def si(self):
      _, nearest_neighbors_indices = torch.min(self.dis_matrix, dim=1)
      si_sum = 0
      for i in tqdm(range(self.n_data), desc="Calculating SI"):
          si_sum += (self.label[i] == self.label[nearest_neighbors_indices[i]]).float().item()
      si = si_sum / self.n_data
      return si

    def compute_high_order_si(self, order):
        try:
            # Sort the distance matrix and get sorted indices
            sorted_distances, sorted_indices = torch.sort(self.dis_matrix, 1)

            repeated_labels = self.label.expand(self.n_data, order)
            sorted_neighbor_labels = self.label[sorted_indices[:, :order]].view(self.n_data, order)

            # Initialize the accumulator for high order separation index
            total_high_order_si = 0

            # Progress tracking
            for idx in tqdm(range(self.n_data), desc="Computing High Order SI"):
                # Check label equality for the first 'order' neighbors and compute the product
                match_labels = (repeated_labels[idx] == sorted_neighbor_labels[idx]).float()
                total_high_order_si += torch.prod(match_labels)

            # Final calculation of high order separation index
            final_high_si = total_high_order_si / self.n_data

            return final_high_si.item()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Insufficient CUDA memory. Consider lowering 'order' or using a device with more GPU memory.")
            else:
                raise e
