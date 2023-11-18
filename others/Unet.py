import os.path
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as op
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True)
])

train_set = datasets.FashionMNIST("D:\\pythonCodes\\numpy\\datasets", train=True, transform=dataset_transforms,
                                  download=True)
test_set = datasets.FashionMNIST("D:\\pythonCodes\\numpy\\datasets", train=False, transform=dataset_transforms,
                                 download=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DownLayer, self).__init__()
        self.DownConv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.DownConv(x)


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UpLayer, self).__init__()
        self.Up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.Conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, y):
        x = self.Up(x)
        diffY = torch.tensor([y.size()[2] - x.size()[2]])  # 得到x和y长和宽的差距
        diffX = torch.tensor([y.size()[3] - x.size()[3]])

        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])  # 将x对y居中，并在四周填充0,目的是为了使x和y的二、三维度一样大，才能进行第一维的cat操作

        x = torch.cat([y, x], dim=1)
        return self.Conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, data_size, kp=None, is_picture=False):
        super(Unet, self).__init__()
        if kp is None:
            kp = [3, 1]
        self.kp = kp
        self.in_channels = in_channels
        self.is_picture = is_picture
        self.out_channels = out_channels
        self.data_size = data_size
        self.in_conv = DoubleConv(in_channels, 64, kernel_size=self.kp[0], padding=self.kp[1])
        self.down1 = DownLayer(64, 128, kernel_size=self.kp[0], padding=self.kp[1])
        self.down2 = DownLayer(128, 256, kernel_size=self.kp[0], padding=self.kp[1])
        self.down3 = DownLayer(256, 512, kernel_size=self.kp[0], padding=self.kp[1])
        self.down4 = DownLayer(512, 1024, kernel_size=self.kp[0], padding=self.kp[1])
        self.up1 = UpLayer(1024, 512, kernel_size=self.kp[0], padding=self.kp[1])
        self.up2 = UpLayer(512, 256, kernel_size=self.kp[0], padding=self.kp[1])
        self.up3 = UpLayer(256, 128, kernel_size=self.kp[0], padding=self.kp[1])
        self.up4 = UpLayer(128, 64, kernel_size=self.kp[0], padding=self.kp[1])
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.fc = nn.Sequential(
            nn.Linear(data_size * data_size * out_channels, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, _img_batch):
        down1 = self.in_conv(_img_batch)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)
        output = self.up1(down5, down4)
        output = self.up2(output, down3)
        output = self.up3(output, down2)
        output = self.up4(output, down1)
        output = self.out_conv(output)
        if self.is_picture is False:
            output = output.view(-1, self.data_size * self.data_size * self.out_channels)
            output = self.fc(output)
        return output


def ClassificationTrain(_net, _Train_Loader, _device, _optimizer, epochs, batch_size=BATCH_SIZE):
    for i in range(epochs):
        print(f" epoch: {i + 1}")
        _net.train()
        score = 0
        count = 0
        min_loss = 1000
        Loss = nn.CrossEntropyLoss()
        time_start = time.time()
        for img_batch, labels_batch in _Train_Loader:
            img_batch, labels_batch = img_batch.to(_device), labels_batch.to(_device)
            _optimizer.zero_grad()
            output = _net(img_batch)
            score += torch.sum(torch.argmax(output, dim=1) == labels_batch)
            loss = Loss(output, labels_batch)
            if loss.item() < min_loss:
                min_loss = loss
                torch.save(_net.state_dict(), 'class_unet.pth')
            loss.backward()
            _optimizer.step()
            count += batch_size
            print('\r',
                  "[" + '=' * (count // 1200) + '>' + ' ' * ((60000 - count) // 1200) + "] {:.2f}%".format(count / 600),
                  end="", flush=True)
        time_end = time.time()
        T = time_end - time_start
        print("\nthe time is {:.4f}s,".format(T), end="")
        print("loss is {:.4f}, acc is {:.4f}%".format(loss.item(), 100 * score.item() / len(_Train_Loader.dataset)))


def PictureTrain(_net, _Train_Loader, _device, _optimizer, epochs, data_size=572, batch_size=BATCH_SIZE):
    for i in range(epochs):
        print(f"epoch: {i + 1}")
        _net.train()
        count = 0
        time_start = time.time()
        min_loss = 10
        BCE = nn.BCEWithLogitsLoss()
        for img_batch, labels_batch in _Train_Loader:
            L = len(_Train_Loader.dataset)
            img_batch, labels_batch = img_batch.to(_device, dtype=torch.float32), labels_batch.to(_device,
                                                                                                  dtype=torch.float32)
            _optimizer.zero_grad()
            output = _net(img_batch)
            # print(output.shape, labels_batch.shape)
            loss = BCE(output, labels_batch)
            loss.backward()
            _optimizer.step()
            count += batch_size
            print("loss is {:.4f}".format(loss.item()))
            if min_loss > abs(loss.item()):
                min_loss = loss.item()
                torch.save(_net.state_dict(), 'unet.pth')
            # print('\r',
            #       "[" + '=' * (count * 50 // L) + '>' + ' ' * ((L - count) * 50 // L) + "] {:.2f}%".format(
            #           count / L * 100),
            #       end="", flush=True)
        time_end = time.time()
        T = time_end - time_start
        print("\nthe time is {:.4f}s\n".format(T), end=" ")


def ClassificationPredict(net, TestLoader, _device):
    net.eval()
    score = 0
    for img_batch, label_batch in TestLoader:
        img_batch, label_batch = img_batch.to(_device, dtype=torch.float32), label_batch.to(_device)
        out = net(img_batch)
        score += torch.sum(torch.argmax(out, dim=1) == label_batch)
        print("the labels are ", end="         ")
        print(np.array(label_batch.data.cpu()))
        print("the predict classes are ", end="")
        print(np.array(torch.argmax(out, dim=1).data.cpu()))
    print("the acc is {:.4f}%".format(100 * score.item() / len(TestLoader.dataset)))
    plt.figure(figsize=(20, 10))
    plt.suptitle('Twelve random images\' prediction')
    for i in range(12):
        rand = np.random.randint(0, 10000, 1).item()
        img, label = test_loader.dataset[rand]
        trans = transforms.Resize((256, 256), antialias=True)
        img = torch.unsqueeze(img, dim=0).to(DEVICE)
        out = MyNet(img)
        img = trans(img)
        img = img.view(256, 256)
        img = img.cpu().numpy()
        plt.subplot(3, 4, i + 1), plt.title('label:' + str(label) + ' predict:' + str(torch.argmax(out, dim=1).item()))
        plt.imshow(img, cmap='gray'), plt.axis('off')
    plt.show()


def PicturePredict(net, pic, _device, is_save=False):
    net.eval()
    count = 0
    root_path = ".\\cell_data\\result"
    trans = transforms.Resize((512, 512), antialias=True)
    if not is_save:
        for img_batch in pic:
            img_batch = img_batch.to(_device, dtype=torch.float32)
            out = net(img_batch)
            out = out.view(512, 512)
            _img = np.array(out.data.cpu())
            _img[_img >= 0.45] = 255
            _img[_img < 0.45] = 0
            img_batch = trans(img_batch)
            img_batch = np.array(img_batch.view(512, 512).data.cpu())
            _img = np.concatenate((img_batch, _img), axis=1)
            cv.imshow(str(count), _img)
            count += 1
            cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        for img_batch in pic:
            img_batch = img_batch.to(_device, dtype=torch.float32)
            out = net(img_batch)
            out = out.view(512, 512)
            _img = np.array(out.data.cpu())
            _img[_img >= 0.5] = 255
            _img[_img < 0.5] = 0
            img_batch = trans(img_batch)
            img_batch = np.array(img_batch.view(512, 512).data.cpu())
            _img = np.concatenate((img_batch, _img), axis=1)
            filename = os.path.join(root_path, str(count))
            filename += ".png"
            cv.imwrite(filename, _img)
            count += 1


def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


if __name__ == '__main__':
    MyNet = Unet(1, 1, 32, kp=[3, 1]).to(DEVICE)
    optimizer = op.Adam(MyNet.parameters())
    state_dict = torch.load('class_unet.pth')
    MyNet.load_state_dict(state_dict)
    # ClassificationTrain(MyNet, train_loader, DEVICE, optimizer, 2, BATCH_SIZE)
    ClassificationPredict(MyNet, test_loader, _device=DEVICE)

