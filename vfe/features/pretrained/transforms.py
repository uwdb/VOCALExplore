import torchvision.transforms as T

def imagenet_normalization():
    # Assumes data is scaled between 0 and 1.
    return T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
