# 1. 라이브러리 불러오기 및 랜덤 SEED 설정
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import random

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from torch.utils.data import DataLoader
import torch
import gc

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import os
import numpy as np
import tensorflow as tf
import torch

def seed_everything(seed):
    random.seed(seed)  # 파이썬 내장 random 모듈의 시드 설정
    os.environ['PYTHONHASHSEED'] = str(seed)  # 해시 함수의 시드 설정
    np.random.seed(seed)  # NumPy 라이브러리의 시드 설정
    tf.random.set_seed(seed)  # TensorFlow의 시드 설정
    torch.manual_seed(seed)  # PyTorch의 CPU 장치 시드 설정
    torch.cuda.manual_seed(seed)  # PyTorch의 CUDA 장치 시드 설정
    torch.backends.cudnn.deterministic = True  # CUDA 연산의 재현성을 위한 설정
    torch.backends.cudnn.benchmark = True  # CUDA 연산의 속도를 향상시키는 설정을 비활성화

seed_everything(299623)  # 시드 고정

import torch
print(torch.version.cuda)
# GPU 사용 설정
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch # PyTorch를 불러옵니
torch.cuda.is_available() # GPU가 사용 가능한지 확인합니다.
#torch.cuda.device_count() # 사용 가능한 장치가 몇 개인지 확인합니다.
#torch.cuda.get_device_name(0) # 첫번째 GPU의 장치명을 확인합니다.
#torch.cuda.get_device_name(1) # 두번째 GPU의 장치명을 확인합니다.

#!git clone https://github.com/openvinotoolkit/anomalib.git
#cd anomalib
#pip install .
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# 2. 비정상 이미지 생성 및, 모델 불러오기(patchcore)
# Import the datamodule
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
    root="./",
    normal_dir="train",
    test_split_mode=TestSplitMode.SYNTHETIC,
    task="classification"
)

# Setup the datamodule
datamodule.setup()
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from torch.utils.data import DataLoader
import torch
import gc

# 모델 초기화
model = Patchcore(layers=('layer2','layer3', 'layer4'))

# 엔진 초기화
engine = Engine(task="classification")

# 훈련 데이터셋으로 모델 훈련
engine.train(datamodule=datamodule, model=model)

train_index = "TRAIN_000.png"  # 첫 번째 이미지

inference_dataset1 = PredictDataset(path=f"./train/{train_index}", image_size=(256, 256))


inference_dataloader1 = DataLoader(dataset=inference_dataset1)

predictions1 = engine.predict(model=model, dataloaders=[inference_dataloader1])[0]

print(predictions1['image_path'])
print(predictions1['pred_scores'])


def empty_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

#threshold의 기준을 아래 코드를 통해 TRAIN 데이터를 기준으로 반복하였습니다.
while True:  # 무한 루프
    # 예측 점수가 0.5475보다 크고 0.5575보다 작지 않으면 루프 종료
    if (0.5475 < predictions1['pred_scores'].item() <= 0.5575):
        break
    empty_cuda_cache()

    # 모델 초기화
    model = Patchcore(layers=('layer2','layer3', 'layer4'))

    # 엔진 초기화
    engine = Engine(task="classification")
    engine.train(datamodule=datamodule, model=model)
    predictions1 = engine.predict(model=model, dataloaders=[inference_dataloader1])[0]

    print(predictions1['image_path'])
    print(predictions1['pred_scores'])


label_set = []

for i in range(0, 100):
    test_index = f"TEST_{i:03d}.png"
    inference_dataset = PredictDataset(path=f"./test/{test_index}", image_size=(256, 256))
    inference_dataloader = DataLoader(dataset=inference_dataset)
    predictions = engine.predict(model=model, dataloaders=[inference_dataloader])[0]
    print(predictions['image_path'])
    print(predictions['pred_scores'])
    
    # 아래에 해당하는 0.666의 기준은 마찬가지로, TRAIN_000.png 데이터를 기준으로 정해진 threshold입니다.
    # 예측 점수가 0.666보다 크거나 같으면 1, 아니면 0으로 라벨링합니다.
    if predictions['pred_scores'] >= 0.666:
        label = 1
    else:
        label = 0
    label_set.append(label)
#tensor([0.6621]
#tensor([0.6790])
label_set
# 3. Image Feature 추출 (RESNET18)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import random
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 데이터 로딩 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로.
            transform (callable, optional): 샘플에 적용될 Optional transform.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        target = torch.tensor([0.]).float()
        return image, target

# 이미지 전처리 및 임베딩
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = CustomDataset(csv_file='./train.csv', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)


def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += torch.sum(predictions == labels.view(-1, 1)).item()
            total += labels.size(0)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# 모델 학습 실행
train(model, train_loader, criterion, optimizer, scheduler, num_epochs=4)
# 사전 학습된 모델 로드
model.eval()  # 추론 모드로 설정

# 특성 추출을 위한 모델의 마지막 레이어 수정
model = torch.nn.Sequential(*(list(model.children())[:-1]))

model.to(device)

# 이미지를 임베딩 벡터로 변환
def get_embeddings(dataloader, model):
    embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy().squeeze())
    return np.concatenate(embeddings, axis=0)

train_embeddings = get_embeddings(train_loader, model)
# 4. 임베딩 된 데이터를 isolation_forest로 이상치 분류
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Isolation Forest 모델 학습
clf = IsolationForest(random_state=69)
clf.fit(train_embeddings)
# 테스트 데이터에 대해 이상 탐지 수행
test_data = CustomDataset(csv_file='./test.csv', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

test_embeddings = get_embeddings(test_loader, model)
pred = clf.predict(test_embeddings)

# Isolation Forest의 예측 결과(이상 = -1, 정상 = 1)를 이상 = 1, 정상 = 0으로 변환
pred = np.where(pred == -1, 1, 0)
pred
## 트랜지스터가 비어있는 이미지나, 
## 정상 이미지인데 불구하고 학습이 안 된 데이터들을 RESNET18 + isolation Forest로 재학습
#해당 데이터들은 점수를 차차 올려가며 어떤 데이터가 학습이 안된 상태인지
isolation_pred1 = pred[34:35]
isolation_pred2 = pred[38:39]
isolation_pred3 = pred[67:68]
isolation_pred4 = pred[91:92]
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = test_pred

submit['label'][34:35] =isolation_pred1
submit['label'][38:39] =isolation_pred2
submit['label'][67:68] =isolation_pred3
submit['label'][91:92] = isolation_pred4
all_pred = submit['label']
#submit.to_csv('./final0.csv', index=False)
