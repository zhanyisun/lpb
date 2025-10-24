import torchvision.transforms as T

def default_transform(img_size=224):
    return T.Compose(
        [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

def get_train_crop_transform(original_img_size, cropped_img_size):
    return T.Compose([
        T.RandomCrop((cropped_img_size, cropped_img_size)),
        T.Resize((original_img_size, original_img_size)),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def get_train_crop_transform_resnet(original_img_size, cropped_img_size):
    return T.Compose([
        T.RandomCrop((cropped_img_size, cropped_img_size)),
    ])

def get_eval_crop_transform(original_img_size, cropped_img_size):
    return T.Compose([
        T.CenterCrop((cropped_img_size, cropped_img_size)),
        T.Resize((original_img_size, original_img_size)),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def get_eval_crop_transform_resnet(original_img_size, cropped_img_size):
    return T.Compose([
        T.CenterCrop((cropped_img_size, cropped_img_size)),
    ])