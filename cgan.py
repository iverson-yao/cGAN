
import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from scipy.linalg import sqrtm
from jittor import nn
from jittor.models.inception import inception_v3
import jittor.transform as transform
import random
import matplotlib.pyplot as plt


# 加载预训练的 Inception v3 模型
inception = inception_v3(pretrained=True)
inception.fc = nn.Identity()  # 去掉全连接层
inception.eval()

# 定义图像预处理
transform_inception = transform.Compose([
    transform.Resize((299, 299)),
    transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_inception_features(images):
    # 确保每个图像都是 (C, H, W) 形状，并且值域在 [-1, 1]
    images = [
        transform_inception(
            Image.fromarray(
                # 将灰度图像转换为 3 通道图像
                np.repeat(((img.numpy().squeeze() + 1) / 2 * 255).astype(np.uint8)[:, :, None], 3, axis=2)
            )
        ) for img in images
    ]
    
    # 将列表中的图像堆叠成一个批次
    images = jt.stack([jt.array(np.array(img)) for img in images])
    
    with jt.no_grad():
        features = inception(images)
    
    return features

def calculate_fid(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    cov_mean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
    return fid


if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs of training') #原来为100
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        self.model = nn.Sequential(*block((opt.latent_dim + opt.n_classes), 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
        img = self.model(gen_input)
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(nn.Linear((opt.n_classes + int(np.prod(img_shape))), 512), nn.LeakyReLU(0.2), nn.Linear(512, 512), nn.Dropout(0.4), nn.LeakyReLU(0.2), nn.Linear(512, 512), nn.Dropout(0.4), nn.LeakyReLU(0.2), nn.Linear(512, 1))

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

# Configure data loader
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)
# 创建测试集的数据加载器
test_dataloader = MNIST(train=False, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=False)


optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

from PIL import Image
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            #img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        #img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    # img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    # img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    elif C==1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.numpy(), "images/%d.png" % batches_done, nrow=n_row)

# ----------
#  Training
# ----------
# 定义要使用的样本数量
num_samples_for_fid = 100

# 随机选择索引
random_indices = random.sample(range(len(test_dataloader) * opt.n_classes), num_samples_for_fid)

# 用于存储损失值和FID分数
d_losses = []
g_losses = []
fid_scores = []


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

        # Configure input
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.sync()
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.sync()
        optimizer_D.step(d_loss)
        if i  % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

    if epoch % 10 == 0:
        # 提取真实图像特征
        real_features = []
        selected_count = 0

        for i, (imgs, _) in enumerate(test_dataloader):
            start_idx = i * opt.batch_size
            end_idx = start_idx + opt.batch_size
            batch_indices = [idx for idx in random_indices if start_idx <= idx < end_idx]
            
            if batch_indices:
                # 只提取被选中的图像
                selected_imgs = imgs[jt.array(batch_indices) - start_idx]
                real_features.append(get_inception_features(selected_imgs))
                selected_count += len(batch_indices)
            
            if selected_count >= num_samples_for_fid:
                break  # 如果已经选择了足够的样本，停止遍历

        real_features = np.concatenate(real_features, axis=0)

        # 生成图像的数量应该与所选的真实图像数量相同
        z = jt.array(np.random.normal(0, 1, (num_samples_for_fid, opt.latent_dim))).float32().stop_grad()
        labels = jt.array(np.random.randint(0, opt.n_classes, num_samples_for_fid)).float32().stop_grad()

        # 生成图像
        gen_imgs = generator(z, labels)

        # 提取生成图像特征
        gen_features = get_inception_features(gen_imgs)

        # 计算 FID
        fid_score = calculate_fid(real_features, gen_features.numpy())
        fid_scores.append(fid_score)
        print(f"[Epoch {epoch}/{opt.n_epochs}] FID: {fid_score:.2f}")

        print("保存模型")
        generator.save("saved_models/generator_last.pkl")
        discriminator.save("saved_models/discriminator_last.pkl")

    # 保存损失值
    d_losses.append(d_loss.data)
    g_losses.append(g_loss.data)

import pickle
# Define Model
generator = Generator()
discriminator = Discriminator()

# Load Parameters
generator.load_parameters(pickle.load(open('saved_models/generator_last.pkl', 'rb')))
discriminator.load_parameters(pickle.load(open('saved_models/discriminator_last.pkl', 'rb')))


number = "2213904"
n_row = len(number)

# Modify z to generate only one row of images
z = jt.array(np.random.normal(0, 1, (n_row, opt.latent_dim))).float32().stop_grad()

# Generate labels for only one row
labels = jt.array([int(num) for num in number]).float32().stop_grad()

# Generate images and save them
gen_imgs = generator(z, labels)
save_image(gen_imgs.numpy(), "result.png", nrow=n_row)

# 绘制损失曲线
plt.figure(figsize=(14, 5))

# D loss 曲线
plt.subplot(1, 2, 1)
plt.plot(d_losses, label="D loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Loss")
plt.legend()

# G loss 曲线
plt.subplot(1, 2, 2)
plt.plot(g_losses, label="G loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 绘制 FID 曲线
plt.figure()
plt.plot(range(0, opt.n_epochs, 10), fid_scores, label="FID")
plt.xlabel("Epoch")
plt.ylabel("FID Score")
plt.title("FID Score over Epochs")
plt.legend()
plt.show()
