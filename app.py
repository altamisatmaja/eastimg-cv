import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch.nn.functional as F


class EASTResNet(nn.Module):
    def __init__(self):
        super(EASTResNet, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  
        self.layer2 = list(resnet.children())[5]  
        self.layer3 = list(resnet.children())[6]  
        self.layer4 = list(resnet.children())[7]  
        
        self.reduce1 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce2 = nn.Conv2d(512, 128, kernel_size=1)
        self.reduce3 = nn.Conv2d(1024, 128, kernel_size=1)
        self.reduce4 = nn.Conv2d(2048, 128, kernel_size=1)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.merge = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.score_map = nn.Conv2d(128, 1, kernel_size=1)
        self.geo_map = nn.Conv2d(128, 4, kernel_size=1)
        self.angle_map = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        c2 = self.layer1(x)  
        c3 = self.layer2(c2)  
        c4 = self.layer3(c3)  
        c5 = self.layer4(c4)  
        
        f5 = self.reduce4(c5)
        f4 = self.up1(f5) + self.reduce3(c4)
        f3 = self.up2(f4) + self.reduce2(c3)
        f2 = self.up3(f3) + self.reduce1(c2)
        
        f = self.merge(f2)
        
        score = torch.sigmoid(self.score_map(f))
        geo = torch.sigmoid(self.geo_map(f)) * 512  
        angle = (torch.sigmoid(self.angle_map(f)) - 0.5) * np.pi  
        
        score = nn.functional.interpolate(score, size=(512, 512), mode='bilinear', align_corners=True)
        geo = nn.functional.interpolate(geo, size=(512, 512), mode='bilinear', align_corners=True)
        angle = nn.functional.interpolate(angle, size=(512, 512), mode='bilinear', align_corners=True)
        
        return score, geo, angle

class EASTTextDataset(Dataset):
    def __init__(self, image_dir, annot_dir, input_size=512):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.input_size = input_size
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annot_path = os.path.join(self.annot_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        
        score_map, geo_map, angle_map = self.generate_ground_truth(annot_path, orig_h, orig_w)
        
        return image, score_map, geo_map, angle_map
    
    def generate_ground_truth(self, annot_path, orig_h, orig_w):
        score_map = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        geo_map = np.zeros((4, self.input_size, self.input_size), dtype=np.float32)
        angle_map = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        
        if not os.path.exists(annot_path):
            return (torch.from_numpy(score_map), 
                    torch.from_numpy(geo_map), 
                    torch.from_numpy(angle_map))
        
        with open(annot_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
                
            
            poly = np.array([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
            poly[:, 0] = poly[:, 0] * (self.input_size / orig_w)
            poly[:, 1] = poly[:, 1] * (self.input_size / orig_h)
            
            
            rect = cv2.minAreaRect(poly)
            center, size, angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            
            cv2.fillPoly(score_map, [box], 1)
            
            
            h, w = self.input_size, self.input_size
            y_coords, x_coords = np.where(score_map > 0)
            for y, x in zip(y_coords, x_coords):
                geo_map[0, y, x] = y  
                geo_map[1, y, x] = w - x  
                geo_map[2, y, x] = h - y  
                geo_map[3, y, x] = x  
                
                
                angle_map[y, x] = angle
        
        return (torch.from_numpy(score_map).unsqueeze(0),  
                torch.from_numpy(geo_map), 
                torch.from_numpy(angle_map).unsqueeze(0))  

class EASTLoss(nn.Module):
    def __init__(self):
        super(EASTLoss, self).__init__()
        self.score_loss = nn.BCELoss(reduction='none')
        self.geo_loss = nn.SmoothL1Loss(reduction='none')
        self.angle_loss = nn.L1Loss(reduction='none')
        
    def forward(self, pred_score, pred_geo, pred_angle, gt_score, gt_geo, gt_angle):
        # Score map loss
        score_mask = (gt_score > 0).float()
        num_positive = torch.sum(score_mask)
        num_negative = torch.sum(1 - score_mask)
        
        # Balance positive and negative samples
        beta = num_negative / (num_positive + num_negative + 1e-5)
        pos_weight = beta / (1 - beta)
        
        # Calculate score loss
        score_loss = self.score_loss(pred_score, gt_score)
        score_loss = score_loss * (pos_weight * score_mask + (1 - score_mask))
        score_loss = torch.mean(score_loss)
        
        # Geometry loss (only on positive samples)
        # Reshape geo maps to [B, 4, H, W]
        pred_geo = pred_geo.permute(0, 3, 1, 2)  # [B, H, W, 4] -> [B, 4, H, W]
        gt_geo = gt_geo.permute(0, 3, 1, 2)      # [B, H, W, 4] -> [B, 4, H, W]
        
        # Expand score mask to match geo dimensions
        geo_score_mask = score_mask.unsqueeze(1)  # [B, 1, H, W]
        
        # Calculate geo loss per channel
        geo_loss = 0
        for i in range(4):  # For each geo channel (top, right, bottom, left)
            channel_loss = self.geo_loss(pred_geo[:, i:i+1], gt_geo[:, i:i+1])
            geo_loss += torch.sum(channel_loss * geo_score_mask) / (num_positive + 1e-5)
        geo_loss /= 4  # Average over 4 channels
        
        # Angle loss (only on positive samples)
        angle_loss = torch.sum(self.angle_loss(pred_angle, gt_angle) * score_mask) / (num_positive + 1e-5)
        
        total_loss = score_loss + 10 * geo_loss + 10 * angle_loss
        return total_loss, score_loss, geo_loss, angle_loss
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = EASTTextDataset('data/ch1_text_localization/training_set/training_set_ch4_training_images', 'data/ch1_text_localization/training_set/ch4_training_localization_transcription_gt')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    model = EASTResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = EASTLoss()
    
    best_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geo_loss = 0
        epoch_angle_loss = 0
        
        for images, score_gt, geo_gt, angle_gt in dataloader:
            images = images.to(device)
            score_gt = score_gt.to(device)
            geo_gt = geo_gt.to(device)
            angle_gt = angle_gt.to(device)
            
            pred_score, pred_geo, pred_angle = model(images)
            
            
            pred_score = pred_score.squeeze(1)  
            pred_angle = pred_angle.squeeze(1)
            geo_gt = geo_gt.permute(0, 2, 3, 1)  
            
            total_loss, score_loss, geo_loss, angle_loss = criterion(
                pred_score, pred_geo, pred_angle,
                score_gt.squeeze(1), geo_gt, angle_gt.squeeze(1))
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_score_loss += score_loss.item()
            epoch_geo_loss += geo_loss.item()
            epoch_angle_loss += angle_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_score_loss = epoch_score_loss / len(dataloader)
        avg_geo_loss = epoch_geo_loss / len(dataloader)
        avg_angle_loss = epoch_angle_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{50}")
        print(f"Total Loss: {avg_loss:.4f} | Score: {avg_score_loss:.4f} | Geo: {avg_geo_loss:.4f} | Angle: {avg_angle_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            export_model(model, 'best_east_model.pth')
            print("Model saved!")

def non_max_suppression(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []
    
    
    polygons = [Polygon(box) for box in boxes]
    areas = [poly.area for poly in polygons]
    
    
    indices = np.argsort(scores)
    keep = []
    
    while len(indices) > 0:
        current = indices[-1]
        keep.append(current)
        indices = indices[:-1]
        
        if len(indices) == 0:
            break
            
        current_poly = polygons[current]
        other_polys = [polygons[i] for i in indices]
        
        
        ious = []
        for other in other_polys:
            intersection = current_poly.intersection(other).area
            union = current_poly.union(other).area
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        
        ious = np.array(ious)
        indices = indices[ious <= threshold]
    
    return keep

def restore_quad(box, score_map, geo_map, angle_map, score_thresh=0.8):
    
    y, x = box
    top = geo_map[0, y, x]
    right = geo_map[1, y, x]
    bottom = geo_map[2, y, x]
    left = geo_map[3, y, x]
    angle = angle_map[y, x]
    
    
    x_min = x - left
    x_max = x + right
    y_min = y - top
    y_max = y + bottom
    
    
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    rot_mat = cv2.getRotationMatrix2D(tuple(center), np.degrees(angle), 1)
    
    points = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    
    
    homog_points = np.hstack((points, np.ones((4, 1))))
    rotated_points = (rot_mat @ homog_points.T).T
    
    return rotated_points.astype(np.int32)

def infer(image_path, model_path='best_east_model.pth', score_thresh=0.8, nms_thresh=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EASTResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    input_img = cv2.resize(image, (512, 512))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score_map, geo_map, angle_map = model(input_tensor)
    
    
    score_map = score_map.squeeze().cpu().numpy()
    geo_map = geo_map.squeeze().cpu().numpy()
    angle_map = angle_map.squeeze().cpu().numpy()
    
    
    y_coords, x_coords = np.where(score_map > score_thresh)
    boxes = []
    scores = []
    
    
    for y, x in zip(y_coords, x_coords):
        quad = restore_quad((y, x), geo_map, angle_map, score_thresh)
        boxes.append(quad)
        scores.append(score_map[y, x])
    
    
    if len(boxes) > 0:
        keep_indices = non_max_suppression(boxes, scores, nms_thresh)
        boxes = [boxes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
    
    
    scale_w = orig_w / 512
    scale_h = orig_h / 512
    scaled_boxes = []
    
    for box in boxes:
        scaled_box = box.copy()
        scaled_box[:, 0] *= scale_w
        scaled_box[:, 1] *= scale_h
        scaled_boxes.append(scaled_box.astype(np.int32))
    
    return scaled_boxes, scores

def test_and_visualize(image_path, model_path='best_east_model.pth'):
    
    boxes, scores = infer(image_path, model_path)
    
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    
    for i, box in enumerate(boxes):
        
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        
        
        center = np.mean(box, axis=0).astype(int)
        cv2.putText(image, f"{scores[i]:.2f}", tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Text Regions (Count: {len(boxes)})")
    plt.axis('off')
    plt.show()
    
    result_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.jpg')
    cv2.imwrite(result_path, image)
    print(f"Result saved to {result_path}")

def export_model(model, filename='east_model.pth'):
    torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    train()

    test_and_visualize('test_image.jpg')