from fastapi import FastAPI
from torch import nn
import pandas as pd
from pydantic import BaseModel
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
from tqdm.notebook import tqdm

app = FastAPI()


class CNNClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNClassificationModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(4),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.out = nn.Sequential(
            nn.Linear(95048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)

        x = self.out(x)

        return x



def load_img(fname):

    img = read_image(fname)
    x = img / 255.

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize image to 224x224 (or the size of your model input)
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            transforms.ColorJitter(brightness=0.2,  # Randomly change brightness, contrast, and saturation
                                   contrast=0.2,
                                   saturation=0.2),
            transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale
        ]
    )

    return transform(x)


def predict(model, test_loader, device):
    """
    Run model inference on test data
    """
    predictions = []
    with torch.no_grad():
        model.eval()  # evaluation mode
        # test_loop = tqdm(enumerate(test_loader, 0), total=len(test_loader), desc="Test")

        for _, inputs in enumerate(test_loader):
            # Write your code here
            # Similar to validation part in training cell
            pred = inputs
            pred = pred.to(device)
            _, predicted = torch.max(model(pred).data, 1)

            # Extend overall predictions by prediction for a batch
            predictions.extend([i.item() for i in predicted])
        return predictions


@app.get("/")
async def root():
    return {"message": "Hello World"}


class DictModel(BaseModel):
    dataframe: dict

@app.post("/predict")
async def prediction(input_data: DictModel):
    model = CNNClassificationModel()
    ckpt = torch.load("../models/best.pt")
    model.load_state_dict(ckpt)
    test_features = pd.DataFrame(input_data.dataframe, columns=['image_id'])
    img_path = "../datasets/archive"
    images = torch.stack(
        [load_img(f"{img_path}/img_align_celeba/test/{item['image_id']}") for _, item in test_features.iterrows()]
    )

    test_loader = DataLoader(images, batch_size=64, shuffle=False)
    predictions = predict(model, test_loader, 'cpu')
    submission_df = pd.DataFrame(columns=['ID', 'predictions'])
    submission_df['ID'] = test_features.index
    submission_df['predictions'] = predictions
    print(submission_df.head())
    return {"output": submission_df.to_dict()}
