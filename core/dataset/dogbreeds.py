from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class DogImageset(Dataset):
    SEP = " " # space char

    def __init__(self, filepath, input_size, preprocessings=[],
                 input_mean=[0.5,0.5,0.5], input_std=[0.5,0.5,0.5]):
        """Dataset wrapper for dog breed images.
        
        Arguments:
            - filepath (str): absolute path of txt file which contains paths_labels pairs
            - preprocessing (arr of fn): preprocessing functions in array 
            - input_mean (arr): input mean in array-like/tuples, must has same length as the input channels(e.g RGB: 3 channels)
            - input_std (arr): input standard dev in array-like/tuples, must has same length as the input channels (e.g RGB: 3 channels) 
        """
        if not os.path.isfile(os.path.abspath(filepath)):
            raise FileNotFoundError("path-label-pairs file (.txt) not found, %s" % filepath)

        # setup attributes
        with open(filepath, 'r') as fp:
            self.path_label_pairs = fh.readlines()
        self.preprocessing_filters = preprocessings
        self.input_mean = input_mean
        self.input_std = input_std
        
        # prepare image preprocessings and transforms
        self.transforms = transforms.Compose(self.preprocessing_filters.extend([
                                              transforms.Resize(self.input_size),
                                              transforms.CenterCrop(self.input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.input_mean, self.input_std)
                                             ]))

    def __getitem__(self, i):
        path, label = self.path_label_pairs[i]
            .strip()
            .split(DogImageset.SEP)
        img = self.transforms(Image.open(path))
        label = int(label)
        
        return (img, label)

    def __len__(self):
        return len(self.path_label_pairs)

