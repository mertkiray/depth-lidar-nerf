import torch.utils.data as data
import torch

class RayDataset(data.Dataset):
    def __init__(self, ray_data, semantic_data=None, use_semantic_data=False):
        super(RayDataset, self).__init__()

        self.rayData = ray_data
        self.length = ray_data.shape[0]
        self.use_semantic_data = use_semantic_data

        if use_semantic_data:
            self.semantic_data = semantic_data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.use_semantic_data:
            return torch.Tensor(self.rayData[index]), torch.tensor(self.semantic_data[index])
        else:
            return torch.Tensor(self.rayData[index])