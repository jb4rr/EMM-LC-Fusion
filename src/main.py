from util.models import VGG16
from util.dataloader import VGG16_Loader
from torchvision import transforms
from torchio.transforms import Resize
from torch.utils.data import DataLoader
import torch
import config

if __name__ == "__main__":
    # Test VGG16 Network

    model = VGG16().float()
    model.load_state_dict(torch.load('./models/VGG16/Best.pth'))
    model.eval()

    test_data = VGG16_Loader('Processed-LIAO-CSV/test_file.csv', preprocessed=True,
                             transform=transforms.Compose([Resize((64, 64, 64))]))
    test_loader = DataLoader(test_data, batch_size=1, num_workers=config.NUM_WORKERS, shuffle=True)

    img, label = test_loader[0]

    out = model(img.float())

    print(f"Prediction: {out}")

    # Test Fusion Network

    pass