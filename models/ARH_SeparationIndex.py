import torch
from tqdm import tqdm

class ARH_SeparationIndex:
    def __init__(self, data, label, normalize=False):
        """
        Initialize the ARH_SeparationIndex class.

        Args:
            data (Tensor): The input features, a tensor of shape (n_data, n_feature).
            label (Tensor): The labels for the data, a tensor of shape (n_data,).
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
        """
        # Set up the device for CUDA support
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize class attributes
        self.normalize = normalize
        self.data = data.to(self.device)
        self.label = label.to(self.device)

        # Normalize data if required
        if normalize:
            self.normalize_data()
            print('Data has been normalized')

        # Adjust labels and calculate necessary statistics
        self.label_min = round(torch.min(self.label).detach().item())
        self.label = (self.label - self.label_min).long()
        self.big_number = 1e10
        # Compute distance matrix
        self.dis_matrix = torch.cdist(self.data, self.data, p=2).fill_diagonal_(self.big_number)
        # Calculate the number of classes, data points, and features
        self.n_class = round(torch.max(self.label).detach().item()) + 1
        self.n_data = self.data.shape[0]
        self.n_feature = self.data.shape[1]

    def normalize_data(self):
        """
        Normalize the data by subtracting the mean and dividing by the standard deviation.
        """
        small_number = 1e-10
        mean_data = torch.mean(self.data, dim=0)
        std_data = torch.std(self.data, dim=0) + small_number
        self.data = (self.data - mean_data) / std_data

    def si(self):
        """
        Calculate the separation index (SI) for the dataset.

        SI measures the proportion of data points having the same label as their nearest neighbor.

        Returns:
            float: The calculated Separation Index (SI).
        """
        _, nearest_neighbors_indices = torch.min(self.dis_matrix, dim=1)
        si_sum = 0
        for i in tqdm(range(self.n_data), desc="Calculating SI"):
            si_sum += (self.label[i] == self.label[nearest_neighbors_indices[i]]).float().item()
        si = si_sum / self.n_data
        return si

    def high_order_si(self, order):
        """
        Calculate the high order separation index for the dataset.

        This index is a stricter version of SI, considering the first 'order' nearest neighbors.

        Args:
            order (int): The order of separation to consider.

        Returns:
            float: The calculated high order separation index.
        """
        try:
            sorted_distances, sorted_indices = torch.sort(self.dis_matrix, 1)
            repeated_labels = self.label.expand(self.n_data, order)
            sorted_neighbor_labels = self.label[sorted_indices[:, :order]].view(self.n_data, order)
            total_high_order_si = 0
            for idx in tqdm(range(self.n_data), desc="Computing High Order SI"):
                match_labels = (repeated_labels[idx] == sorted_neighbor_labels[idx]).float()
                total_high_order_si += torch.prod(match_labels)
            final_high_si = total_high_order_si / self.n_data
            return final_high_si.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Insufficient CUDA memory. Consider lowering 'order' or using a device with more GPU memory.")
            else:
                raise e

    def anti_si(self, order):
        """
        Calculate the anti-separation index (Anti-SI) for the dataset.

        This index measures the proportion of data points having different labels from their 'order' nearest neighbors.

        Args:
            order (int): The order of separation to consider.

        Returns:
            float: The calculated anti-separation index.
        """
        try:
            sorted_dist, sorted_indices = torch.sort(self.dis_matrix, dim=1)
            if self.label.dim() == 1:
                labels = self.label.unsqueeze(1)
            else:
                labels = self.label
            expanded_labels = labels.expand(self.n_data, order)
            nearest_neighbor_labels = labels[sorted_indices[:, :order]].view(self.n_data, order)
            total_anti_si = 0
            for i in tqdm(range(self.n_data), desc="Calculating Anti-SI"):
                label_difference = 1 - (expanded_labels[i] == nearest_neighbor_labels[i]).float()
                total_anti_si += torch.prod(label_difference)
            final_anti_si = total_anti_si / self.n_data
            return final_anti_si.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Try reducing 'order' or using a device with more memory.")
            else:
                raise e

    def soft_order_si(self, order):
        """
        Calculate the soft order separation index (Soft-SI) for the dataset.

        This index provides a less strict measure of separation, considering matching labels among 'order' nearest neighbors.

        Args:
            order (int): The order of separation to consider.

        Returns:
            float: The calculated soft order separation index.
        """
        try:
            sorted_distances, neighbor_indices = torch.sort(self.dis_matrix, dim=1)
            if self.label.dim() == 1:
                labels_reshaped = self.label.unsqueeze(1)
            else:
                labels_reshaped = self.label
            expanded_labels = labels_reshaped.expand(self.n_data, order)
            neighbor_labels = labels_reshaped[neighbor_indices[:, :order]].view(self.n_data, order)
            total_soft_si = 0
            for i in tqdm(range(self.n_data), desc="Calculating Soft Order SI"):
                matching_labels_count = (expanded_labels[i] == neighbor_labels[i]).sum()
                total_soft_si += matching_labels_count.float() / order
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

        CSI measures the proportion of data points closest to the mean of their respective classes.
        It's a faster computation method, especially suitable for datasets where each class forms a unique and normal distribution.

        Returns:
            float: The calculated Center-based Separation Index (CSI).
        """
        try:
            class_centers = torch.stack([
                self.data[self.label.squeeze() == cls].mean(dim=0)
                for cls in tqdm(range(self.n_class), desc="Calculating Class Centers")
            ])
            distances_to_centers = torch.cdist(self.data, class_centers, p=2)
            nearest_center_labels = torch.argmin(distances_to_centers, dim=1)
            csi = torch.sum(nearest_center_labels == self.label.squeeze()).float() / self.n_data
            return csi.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Try reducing the dataset size or using a device with more GPU memory.")
                return None
            else:
                raise e
