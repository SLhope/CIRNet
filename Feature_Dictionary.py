import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.vgg19(pretrained=True).features.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def global_average_pooling(features):
    return features.mean(dim=(2, 3))


def interpolate_to_512_classes(features, global_features):
    features_diff = features 
    diff_abs = torch.abs(features_diff).sum(dim=1) 
    
    _, sorted_indices = torch.sort(diff_abs, descending=True)

    num_samples = len(sorted_indices)
    print(diff_abs.shape)
    print(num_samples)
    
    group_size = num_samples // 512
    
    averaged_features = []
    
    for i in range(512):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size if i < 511 else num_samples
        
        group_indices = sorted_indices[start_idx:end_idx]
        
        group_avg = features[group_indices].mean(dim=0)
        averaged_features.append(group_avg)
    
    print(len(averaged_features))
    print(averaged_features[0].shape)
    return torch.stack(averaged_features)


def read_image_and_masks(image_path, shadow_mask_path, non_shadow_mask_path, background_mask_path):
    image = Image.open(image_path).convert('RGB')
    shadow_mask = Image.open(shadow_mask_path).convert('L')
    non_shadow_mask = Image.open(non_shadow_mask_path).convert('L')
    background_mask = Image.open(background_mask_path).convert('L')
    return image, shadow_mask, non_shadow_mask, background_mask

def extract_features(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    masked_image = np.zeros_like(image)
    for i in range(3):
        masked_image[:, :, i] = image[:, :, i] * (mask / 255.0)
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    
    masked_image = transform(masked_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(masked_image)
    return features

def main():
    image_dir = '/home/yj/Dataset/patch_image'
    shadow_mask_dir = '/home/yj/Dataset/patch_shadow_image'
    non_shadow_mask_dir = '/home/yj/Dataset/patch_non_shadow_image'
    background_mask_dir = '/home/yj/Dataset/patch_background_image'
    
    non_shadow_features_list = []
    background_features_list = []
    
    for i in range(2):
        image_path = os.path.join(image_dir, f'image_{i}.jpg')
        shadow_mask_path = os.path.join(shadow_mask_dir, f'shadow_mask_{i}.png')
        non_shadow_mask_path = os.path.join(non_shadow_mask_dir, f'non_shadow_mask_{i}.png')
        background_mask_path = os.path.join(background_mask_dir, f'bg_mask_{i}.png')
        
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            file_base_name = os.path.splitext(filename)[0]
            
            image_path = os.path.join(image_dir, filename)
            shadow_mask_path = os.path.join(shadow_mask_dir, f'{file_base_name}.png')
            non_shadow_mask_path = os.path.join(non_shadow_mask_dir, f'{file_base_name}.png')
            background_mask_path = os.path.join(background_mask_dir, f'{file_base_name}.png')

            image, _, non_shadow_mask, background_mask = read_image_and_masks(image_path, shadow_mask_path, non_shadow_mask_path, background_mask_path)
            
            non_shadow_features = extract_features(image, non_shadow_mask)
            background_features = extract_features(image, background_mask)
            
            non_shadow_features = global_average_pooling(non_shadow_features).squeeze(0)
            background_features = global_average_pooling(background_features).squeeze(0)
            
            non_shadow_features_list.append(non_shadow_features)
            background_features_list.append(background_features)
    
    non_shadow_features_tensor = torch.stack(non_shadow_features_list)
    background_features_tensor = torch.stack(background_features_list)
    
    global_shadow_features = non_shadow_features_tensor.mean(dim=0)
    global_background_features = background_features_tensor.mean(dim=0)
    
    non_shadow_features_diff = non_shadow_features_tensor - global_shadow_features
    background_features_diff = background_features_tensor - global_background_features
    
    interpolated_non_shadow_features = interpolate_to_512_classes(non_shadow_features_diff, global_shadow_features)
    interpolated_background_features = interpolate_to_512_classes(background_features_diff, global_background_features)
    
    torch.save(interpolated_non_shadow_features, '/home/yj/Dataset/patch_dict/interpolated_non_shadow_features.pth')
    torch.save(interpolated_background_features, '/home/yj/Dataset/patch_dict/interpolated_background_features.pth')

if __name__ == '__main__':
    main()