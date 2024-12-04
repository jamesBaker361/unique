import diffusers
from datasets import load_dataset

from transformers import AutoImageProcessor
import evaluate

accuracy = evaluate.load("accuracy")

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
composed_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])



def transforms(examples):
    examples["pixel_values"] = [composed_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

data=load_dataset("jlbaker361/real-fake-league")

split_dataset = data['train'].train_test_split(test_size=0.2, seed=42)

# Step 3: Access the split datasets
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

split_dataset=split_dataset.with_transform(transforms)

