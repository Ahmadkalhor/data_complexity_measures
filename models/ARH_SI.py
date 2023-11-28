import torch

class ARH_SeparationIndex:
    """
    This class provides an implementation of various Separation Index (SI) based methods
    to analyze the separability of data points in classification problems.

    The class supports calculations of the standard Separation Index, high-order Separation Index,
    and their variants, offering insights into the complexity and characteristics of classification datasets.

    Attributes:
        data (Tensor): A tensor of shape (n_data, n_feature) representing input feature points.
        label (Tensor): A tensor of shape (n_data, 1) representing labels of the data points.
        normalize (bool): A flag to indicate whether the input data should be normalized.
        batch_size (int): The size of each batch to be processed, for efficient memory usage.
    """

    def __init__(self, data, label, normalize=False, batch_size=1000):
        """
        Initializes the ARH_SeparationIndex object with data, labels, and optional normalization.

        Args:
            data (Tensor): Input features tensor of shape (n_data, n_feature).
            label (Tensor): Labels tensor of shape (n_data, 1).
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
            batch_size (int, optional): Size of batches for processing large datasets. Defaults to 1000.
        """
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
        # self.n_class = self.label.max().item() + 1
        self.n_class = int(self.label.max().item() + 1)
        self.n_data = self.data.shape[0]
        self.n_feature = self.data.shape[1]

    def si(self):
        """
        Calculates the standard Separation Index (SI) for the dataset.

        SI measures the proportion of data points that have their nearest neighbor
        belonging to the same class. It is a normalized measure ranging between 0 and 1,
        where higher values indicate better separability.

        Returns:
            float: The calculated Separation Index (SI).
        """
        values, indices = torch.min(self.dis_matrix, 1)
        si_data = (self.label[indices] == self.label)
        si = si_data.sum().float() / self.n_data
        return si

    def si_data(self):
        """
        Calculates the Separation Index (SI) for each individual data point in the dataset.

        This method provides a granular view of separability, where the SI for each point
        is determined based on its nearest neighbor.

        Returns:
            Tensor: A tensor of shape (n_data,) containing the SI for each data point.
        """
        values, indices = torch.min(self.dis_matrix, 1)
        si_data = (self.label[indices] == self.label).float()
        return si_data

    def high_order_si(self, order):
        """
        Calculates a high-order variant of the Separation Index (SI) for the dataset.

        This method extends the concept of SI by considering the first 'order' nearest neighbors
        for each data point and checks if they belong to the same class.

        Args:
            order (int): The number of nearest neighbors to consider.

        Returns:
            float: The calculated high-order Separation Index.
        """
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
        """
        Calculates the high-order Separation Index (SI) for each data point in the dataset.

        Similar to `high_order_si`, but provides the SI for each individual data point
        based on its 'order' nearest neighbors.

        Args:
            order (int): The number of nearest neighbors to consider.

        Returns:
            Tensor: A tensor of shape (n_data,) containing the high-order SI for each data point.
        """
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

    def anti_si(self, order):
          """
          Calculates the anti Separation Index (anti-SI) for the dataset.

          This method evaluates how often data points have labels different from their 'order'
          nearest neighbors, providing a normalized score between 0 and 1.

          Args:
              order (int): The number of nearest neighbors to consider.

          Returns:
              float: The calculated anti Separation Index.
          """
          anti_si_sum = 0
          for i in range(0, self.n_data, self.batch_size):
              batch = self.data[i:i + self.batch_size]
              dis_matrix = torch.cdist(batch, self.data, p=2)
              dis_matrix.fill_diagonal_(self.big_number)
              _, arg_sort = torch.sort(dis_matrix, 1)
              sorted_labels = self.label[arg_sort[:, :order]]
              comp_label = 1 - (self.label[i:i + self.batch_size].unsqueeze(1) == sorted_labels).float()
              anti_si_sum += comp_label.prod(2).sum().item()

          anti_si = anti_si_sum / self.n_data
          return anti_si
    
    def anti_si_data(self, order):
      """
      Calculates the anti Separation Index (anti-SI) for each data point in the dataset.

      This method provides a granular view of anti-separability, where the anti-SI for each point
      is determined based on its 'order' nearest neighbors.

      Args:
          order (int): The number of nearest neighbors to consider.

      Returns:
          Tensor: A tensor of shape (n_data,) containing the anti-SI for each data point.
      """
      anti_si_data = torch.zeros(self.n_data, device=self.device)
      for i in range(0, self.n_data, self.batch_size):
          batch = self.data[i:i + self.batch_size]
          dis_matrix = torch.cdist(batch, self.data, p=2)
          dis_matrix.fill_diagonal_(self.big_number)
          _, arg_sort = torch.sort(dis_matrix, 1)
          sorted_labels = self.label[arg_sort[:, :order]]
          comp_label = 1 - (self.label[i:i + self.batch_size].unsqueeze(1) == sorted_labels).float()
          anti_si_batch = comp_label.prod(2).view(-1)  # Ensure it is one-dimensional
          batch_end = min(i + self.batch_size, self.n_data)
          anti_si_data[i:batch_end] = anti_si_batch[:batch_end - i]

      return anti_si_data

    def center_si(self):
        """
        Calculates the center-based Separation Index (CSI) for the dataset.

        CSI is a faster computation method for datasets where each class forms a unique and normal distribution.
        It measures the proportion of data points closest to the mean of their respective classes.

        Returns:
            float: The calculated Center-based Separation Index (CSI).
        """
        class_centers = torch.stack([self.data[self.label.squeeze() == cls].mean(dim=0) for cls in range(self.n_class)])
        distances_to_centers = torch.cdist(self.data, class_centers, p=2)
        nearest_center_labels = torch.argmin(distances_to_centers, dim=1)
        csi = torch.sum(nearest_center_labels == self.label.squeeze()).float() / self.n_data

        return csi

    def center_si_data(self):
        """
        Calculates the center-based Separation Index (CSI) for each data point in the dataset.

        This method provides an individual CSI score for each data point, indicating whether it is
        closest to the mean of its respective class.

        Returns:
            Tensor: A tensor of shape (n_data,) containing the CSI for each data point.
        """
        class_centers = torch.stack([self.data[self.label.squeeze() == cls].mean(dim=0) for cls in range(self.n_class)])
        distances_to_centers = torch.cdist(self.data, class_centers, p=2)
        nearest_center_labels = torch.argmin(distances_to_centers, dim=1)
        csi_data = (nearest_center_labels == self.label.squeeze()).float()

        return csi_data  
