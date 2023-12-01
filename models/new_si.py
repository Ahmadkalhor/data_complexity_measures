import torch
from tqdm import tqdm

class ARH_SeparationIndex:
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

    def high_order_si(self, order):
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

    def anti_si(self, order):
        try:
            # Sort the distance matrix and get the indices of sorted neighbors
            sorted_dist, sorted_indices = torch.sort(self.dis_matrix, dim=1)

            # Ensure self.label is a 2D tensor [n_data, 1]
            if self.label.dim() == 1:
                labels = self.label.unsqueeze(1)
            else:
                labels = self.label

            expanded_labels = labels.expand(self.n_data, order)
            nearest_neighbor_labels = labels[sorted_indices[:, :order]].view(self.n_data, order)

            # Initialize the accumulator for the anti separation index
            total_anti_si = 0

            for i in tqdm(range(self.n_data), desc="Calculating Anti-SI"):
                # Compute the difference in labels for the first 'order' neighbors
                label_difference = 1 - (expanded_labels[i] == nearest_neighbor_labels[i]).float()
                total_anti_si += torch.prod(label_difference)

            # Calculate the anti separation index
            final_anti_si = total_anti_si / self.n_data

            return final_anti_si.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Try reducing 'order' or using a device with more memory.")
            else:
                raise e

    def soft_order_si(self, order):
      try:
          # Sort the distance matrix and get the indices of sorted neighbors
          sorted_distances, neighbor_indices = torch.sort(self.dis_matrix, dim=1)

          # Ensure self.label is 2D for broadcasting
          if self.label.dim() == 1:
              labels_reshaped = self.label.unsqueeze(1)
          else:
              labels_reshaped = self.label

          # Expand the labels to match the number of neighbors
          expanded_labels = labels_reshaped.expand(self.n_data, order)
          neighbor_labels = labels_reshaped[neighbor_indices[:, :order]].view(self.n_data, order)

          # Initialize the accumulator for the soft separation index
          total_soft_si = 0

          # Use tqdm for progress tracking
          for i in tqdm(range(self.n_data), desc="Calculating Soft Order SI"):
              # Count matching labels for the first 'order' neighbors and compute the soft score
              matching_labels_count = (expanded_labels[i] == neighbor_labels[i]).sum()
              total_soft_si += matching_labels_count.float() / order

          # Calculate the soft separation index
          final_soft_si = total_soft_si / self.n_data

          return final_soft_si.item()

      except RuntimeError as e:
          if "out of memory" in str(e):
              print("CUDA out of memory. Try reducing 'order' or using a device with more memory.")
          else:
              raise e 
    def center_si(self):
        """
        Calculates the center-based Separation Index (CSI) for the dataset.

        CSI is a faster computation method for datasets where each class forms a unique and normal distribution.
        It measures the proportion of data points closest to the mean of their respective classes.

        Returns:
            float: The calculated Center-based Separation Index (CSI).
        """
        try:
            # Calculate class centers with a progress bar
            class_centers = torch.stack([
                self.data[self.label.squeeze() == cls].mean(dim=0)
                for cls in tqdm(range(self.n_class), desc="Calculating Class Centers")
            ])

            # Compute distances from each data point to the class centers
            distances_to_centers = torch.cdist(self.data, class_centers, p=2)

            # Identify the nearest center for each data point
            nearest_center_labels = torch.argmin(distances_to_centers, dim=1)

            # Calculate the CSI
            csi = torch.sum(nearest_center_labels == self.label.squeeze()).float() / self.n_data

            return csi.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Try reducing the dataset size or using a device with more GPU memory.")
                return None
            else:
                raise e
            
