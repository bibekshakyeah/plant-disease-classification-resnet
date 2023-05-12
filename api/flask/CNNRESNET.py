
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import torchvision.transforms as transforms
import io

# for moving data into GPU (if available)
def get_default_device():
    # """Pick GPU if available, else CPU"""
    # if torch.cuda.is_available:
    #     return torch.device("cuda")
    # else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# base class for the model
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
# Architecture for training

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out            
classes=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Rice___BrownSpot',
 'Rice___Hispa',
 'Rice___LeafBlast',
 'Rice___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


def predict_image(img_bytes, model):
    """Converts image bytes to a PIL image, applies minimal preprocessing, and returns the predicted class
        with the highest probability and the accuracy percentage.

    Args:
        img_bytes (bytes): Image data in bytes format.
        model (nn.Module): Pretrained model for prediction.

    Returns:
        Tuple[str, float]: Predicted class label and accuracy percentage.
    """
    # Convert image bytes to PIL Image
    img = Image.open(img_bytes).convert('RGB')

    # Resize the image to the expected input size of the model
    img = img.resize((256, 256))

    # Convert PIL Image to tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img)

    # Move the tensor to the appropriate device
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Convert to a batch of 1
    img_tensor = img_tensor.unsqueeze(0)

    # Get predictions from the model
    outputs = model(img_tensor)

    # Pick index with highest probability
    _, preds = torch.max(outputs, dim=1)

    # Retrieve the class label
    predicted_class = classes[preds.item()]

    # Calculate the accuracy percentage
    softmax_probs = F.softmax(outputs, dim=1)
    accuracy_percent = round(softmax_probs[0][preds.item()].item() * 100, 2)

    # Return the predicted class label and accuracy percentage
    return {
        'predictClass':predicted_class, 'accuracyPercentage':accuracy_percent
    }







PATH = 'plant-disease-model.pth'  
device = get_default_device()
print(device)
state_dict = torch.load(PATH,map_location=device)
# create the model architecture
model = ResNet9(3, len(classes))
model.load_state_dict(state_dict)
model.to(device)

# move the model to the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

model.eval()
# test_dir = "/content/drive/MyDrive/ml/input/test"
# test_images = sorted(os.listdir(test_dir + '/test')) # since images in test folder are in alphabetical order

# test = ImageFolder(test_dir, transform=transforms.ToTensor())
# for i, (img, label) in enumerate(test):
#     print('Label:', test_images[i], ', Predicted:', predict_image(img, model))