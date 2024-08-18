transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset = RPNDataset('datasets/dataset_v2', transform=transform, resize=ResizeWithPad(224))

total_size = len(dataset)
train_size = int(0.80 * total_size)
val_size = int(0.10 * total_size)
test_size = total_size - train_size - val_size

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Set the batch size
batch_size = 128

# Create the DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Create the DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

image_path = 'datasets/dataset_v2/1.jpg'
boxes, scores = inference(model, image_path, device, transform, ResizeWithPad(224))
# %%
image = Image.open(image_path).convert("RGB")
image = ResizeWithPad(224)(image)

target = boxes.tolist()


draw = Draw(image)
for bb in target:
    draw.rectangle(bb, outline=(255, 0, 0), width=2)
image.show()