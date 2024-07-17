import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def calculate_fid(model, img1, img2):
    def get_activation_statistics(img):
        activations = model(img)
        act = activations[-1].view(activations[-1].size(1), -1).detach().cpu().numpy()
        mu = np.mean(act, axis=1)
        sigma = np.cov(act)
        return mu, sigma
    
    mu1, sigma1 = get_activation_statistics(img1)
    mu2, sigma2 = get_activation_statistics(img2)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 356

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

day_folder = "content"
night_folder = "style"
gen_folder = "Gen_Images"

# if not os.path.exists(gen_folder):
#     os.makedirs(gen_folder)

day_images = sorted(os.listdir(day_folder))
night_images = sorted(os.listdir(night_folder))
model = VGG().to(device).eval()

total_steps =1500
learning_rate = 0.01

alpha = 1
beta = 0.01
print_every = 300

FID_cg = []
FID_sg = []

loss_at_last_step = []  # Added: List to store loss at the last step
all_loss_values = [] # List to store all loss values for combined plot

for day_img_name, night_img_name in zip(day_images, night_images):
    original_img = load_image(os.path.join(day_folder, day_img_name))
    style_img = load_image(os.path.join(night_folder, night_img_name))
    generated = original_img.clone().requires_grad_(True)
    
    optimizer = optim.Adam([generated], lr=learning_rate)
    
    loss_values = []

    for step in range(1,total_steps+1):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)
            G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
            A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_values.append(total_loss.item())
        
        if step % print_every == 0:
            print(f"Step [{step}/{total_steps}], Loss: {total_loss.item():.4f}")

    # Store the loss at the last step for each image
    loss_at_last_step.append(loss_values[-1])
    
    # Append the loss values for the current image to the combined list
    all_loss_values.append(loss_values)
    
    gen_image_name = f"{os.path.splitext(day_img_name)[0]}_gen.png"
    save_image(generated, os.path.join(gen_folder, gen_image_name))
    
    fid_cg = calculate_fid(model, original_img, generated)
    fid_sg = calculate_fid(model, style_img, generated)
    
    FID_cg.append(fid_cg)
    FID_sg.append(fid_sg)

    # Plotting the loss values
    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch for {day_img_name}')
    plt.savefig(f"{gen_folder}/{os.path.splitext(day_img_name)[0]}_loss_plot.png")
    plt.close()

print("FID scores between content and generated images: ", FID_cg)
print("FID scores between style and generated images: ", FID_sg)

# Calculate and print average FID scores
average_fid_cg = np.mean(FID_cg)
average_fid_sg = np.mean(FID_sg)
print("Average FID score between content and generated images: ", average_fid_cg)
print("Average FID score between style and generated images: ", average_fid_sg)

# Added: Calculate and print average of all losses of all images at the last step
average_loss = np.mean(loss_at_last_step)
print("Average loss at the last step for all images: ", average_loss)

# Plotting combined loss values
plt.figure()
for i, loss_values in enumerate(all_loss_values):
    plt.plot(loss_values, label=f'Image {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Combined Loss vs Epoch for All Images')
plt.legend()
plt.savefig(f"{gen_folder}/combined_loss_plot.png")
plt.close()