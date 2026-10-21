import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
from tqdm import tqdm
import torchmetrics
import csv

# ==========================================
# 0. НАСТРОЙКИ И КОНСТАНТЫ
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 128       # Оптимально для 4GB VRAM на 32x32
NUM_WORKERS = 0        # КРИТИЧНО для 8GB RAM!
IMAGE_SIZE = 32
LATENT_DIM = 128
NUM_EPOCHS = 30        # 30 эпох на каждое обучение (достаточно для сходимости DCGAN на CIFAR)
TOTAL_FAKE_TO_GEN = 50000 # Размер синтетического датасета для следующего поколения

# Папки
REAL_DATA_DIR = "./data/cifar10"
FAKE_DATA_ROOT = "./fake_datasets"
RESULTS_DIR = "./results"
os.makedirs(FAKE_DATA_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Пропорции для параллельных цепочек: 100%, 75%, 50%, 25%, 0% реальных данных
# Можешь убрать 0.75, если хочешь ускорить процесс, оставив 1.0, 0.5, 0.25, 0.0
MIXING_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0] 
NUM_GENERATIONS = 3  # Gen 0, Gen 1, Gen 2, Gen 3

# ==========================================
# 1. АРХИТЕКТУРА DCGAN (Стандартная для 32x32)
# ==========================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),        nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),        nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),         nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),           nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),         nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),       nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),      nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),      nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),        nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# ==========================================
# 2. КАСТОМНЫЙ DATASET ДЛЯ СМЕШИВАНИЯ
# ==========================================
class MixedDataset(Dataset):
    def __init__(self, real_dataset, fake_paths, real_ratio, total_size=50000, transform=None):
        self.real_dataset = real_dataset
        self.fake_paths = fake_paths
        self.real_ratio = real_ratio
        self.total_size = total_size
        self.transform = transform

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if random.random() < self.real_ratio:
            img, _ = self.real_dataset[idx % len(self.real_dataset)]
        else:
            fake_idx = idx % len(self.fake_paths)
            img = Image.open(self.fake_paths[fake_idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(1)

# ==========================================
# 3. ФУНКЦИИ ОБУЧЕНИЯ, ГЕНЕРАЦИИ И ОЦЕНКИ
# ==========================================
def train_gan(dataloader, num_epochs, model_name="model"):
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            # Update D
            netD.zero_grad()
            label = torch.ones(b_size, device=device)
            errD_real = criterion(netD(real_cpu), label)
            errD_real.backward()

            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            errD_fake = criterion(netD(fake.detach()), label)
            errD_fake.backward()
            optimizerD.step()

            # Update G
            netG.zero_grad()
            label.fill_(1)
            errG = criterion(netD(fake), label)
            errG.backward()
            optimizerG.step()
            
    return netG

def generate_images(netG, num_images, save_dir):
    netG.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_images, BATCH_SIZE), desc="Генерация"):
            current_batch = min(BATCH_SIZE, num_images - i)
            noise = torch.randn(current_batch, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            
            for j in range(current_batch):
                img = fake[j].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2.0
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(save_dir, f"fake_{i+j:05d}.png"))
    netG.train()

def calculate_fid(real_dataloader, fake_dir, num_samples=10000):
    print("  -> Расчет FID...")
    fid = torchmetrics.image.FrechetInceptionDistance(feature=2048).to(device)
    
    real_count = 0
    for imgs, _ in real_dataloader:
        imgs = (imgs * 255).to(torch.uint8).to(device)
        fid.update(imgs, real=True)
        real_count += imgs.size(0)
        if real_count >= num_samples:
            break
            
    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.png')]
    random.shuffle(fake_paths)
    fake_paths = fake_paths[:num_samples]
    transform = transforms.Compose([transforms.ToTensor()])
    
    for i in tqdm(range(0, len(fake_paths), BATCH_SIZE), desc="  FID Fake"):
        batch_paths = fake_paths[i:i+BATCH_SIZE]
        batch_imgs = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
        batch_imgs = torch.stack(batch_imgs)
        batch_imgs = (batch_imgs * 255).to(torch.uint8).to(device)
        fid.update(batch_imgs, real=False)
        
    return fid.compute().item()

# ==========================================
# 4. ГЛАВНЫЙ ЦИКЛ: ПАРАЛЛЕЛЬНЫЕ РЕКУРСИВНЫЕ ЦЕПОЧКИ
# ==========================================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("Загрузка CIFAR-10 (реальные данные)...")
    real_dataset = CIFAR10(root=REAL_DATA_DIR, train=True, download=True, transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Словарь для хранения всех результатов: { "Chain_100R": { "Gen0": 15.2, "Gen1": 16.1 ... } }
    all_results = {}
    
    # CSV файл для удобного импорта в Excel/Python для построения графиков
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Chain", "Generation", "Real_Ratio", "FID_Score"])

    for ratio in MIXING_RATIOS:
        ratio_pct = int(ratio * 100)
        chain_name = f"Chain_{ratio_pct}R"
        print(f"\n{'='*60}")
        print(f" ЗАПУСК ЦЕПОЧКИ: {chain_name} ({ratio_pct}% Real / {100-ratio_pct}% Fake)")
        print(f"{'='*60}")
        
        all_results[chain_name] = {}
        current_fake_paths = [] # Пути к синтетике текущего поколения в ЭТОЙ цепочке

        for gen in range(NUM_GENERATIONS + 1):
            gen_name = f"Gen{gen}"
            print(f"\n[{chain_name}] Обучение {gen_name}...")
            
            # 1. Подготовка DataLoader
            if gen == 0:
                # Gen 0 ВСЕГДА обучается на 100% реальных данных, независимо от цепочки
                current_dataloader = real_dataloader
                print("  -> Использование 100% реальных данных (Baseline)")
            else:
                # Gen 1+ обучается на смеси, заданной пропорцией chain_name
                mixed_dataset = MixedDataset(
                    real_dataset=real_dataset, 
                    fake_paths=current_fake_paths, 
                    real_ratio=ratio, 
                    total_size=TOTAL_FAKE_TO_GEN, 
                    transform=transform
                )
                current_dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
                print(f"  -> Использование смеси: {ratio_pct}% Real / {100-ratio_pct}% Fake от предыдущего поколения")

            # 2. Обучение
            netG = train_gan(current_dataloader, NUM_EPOCHS, model_name=f"{chain_name}_{gen_name}")
            
            # 3. Генерация датасета для СЛЕДУЮЩЕГО поколения в этой цепочке
            gen_fake_dir = os.path.join(FAKE_DATA_ROOT, chain_name, gen_name)
            print(f"[{chain_name}] Генерация {TOTAL_FAKE_TO_GEN} изображений для следующего шага...")
            generate_images(netG, TOTAL_FAKE_TO_GEN, gen_fake_dir)
            
            # Обновляем пути к фейкам для следующей итерации этой цепочки
            current_fake_paths = [os.path.join(gen_fake_dir, f) for f in os.listdir(gen_fake_dir) if f.endswith('.png')]
            
            # 4. Оценка FID (всегда сравниваем с ИСХОДНЫМИ реальными данными, чтобы видеть деградацию)
            fid_score = calculate_fid(real_dataloader, gen_fake_dir, num_samples=10000)
            all_results[chain_name][gen_name] = fid_score
            print(f"[{chain_name}] {gen_name} FID Score: {fid_score:.2f}")
            
            # Сохраняем в CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([chain_name, gen_name, ratio, fid_score])
            
            # Сохраняем веса модели (опционально, можно закомментировать для экономии места на диске)
            # torch.save(netG.state_dict(), os.path.join(RESULTS_DIR, f"{chain_name}_{gen_name}_weights.pth"))
            
            # 5. Очистка памяти перед следующим поколением
            del netG
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # 5. ФИНАЛЬНЫЙ ОТЧЕТ
    # ==========================================
    print("\n" + "="*70)
    print(" ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО!")
    print("="*70)
    print("Сводная таблица FID (чем меньше, тем лучше):")
    print(f"{'Поколение':<10} | {'100% Real':<10} | {'75% Real':<10} | {'50% Real':<10} | {'25% Real':<10} | {'0% Real':<10}")
    print("-" * 70)
    
    for gen in range(NUM_GENERATIONS + 1):
        gen_name = f"Gen{gen}"
        row = f"{gen_name:<10} | "
        for ratio in MIXING_RATIOS:
            chain_name = f"Chain_{int(ratio*100)}R"
            score = all_results[chain_name].get(gen_name, "N/A")
            if isinstance(score, float):
                row += f"{score:<10.2f} | "
            else:
                row += f"{str(score):<10} | "
        print(row)
    
    print(f"\nВсе данные сохранены в: {csv_path}")
    print(f"Сгенерированные изображения находятся в: {FAKE_DATA_ROOT}")