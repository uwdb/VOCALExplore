from vfe import datasets
from vfe import features

def get_extractor(model, return_class=False, **kwargs):
    if not return_class:
        assert 'features_dir' in kwargs

    if model.startswith('resnet18'):
        extractor_cls = features.pretrained.models.Resnet18Extractor
    elif model.startswith('inception_v3'):
        extractor_cls = features.pretrained.models.InceptionV3Extractor
    elif model.startswith('r3d_18_ap_mean'):
        if 'stride32' in model:
            extractor_cls = features.pretrained.models.Resnet3d18MeanAPStride32Extractor
        else:
            extractor_cls = features.pretrained.models.Resnet3d18MeanAPExtractor
    elif model.startswith('r3d_18'):
        extractor_cls = features.pretrained.models.Resnet3d18Extractor
    elif model.startswith('i3d_mean'):
        extractor_cls = features.pretrained.models.I3DMeanExtractor
    elif model.startswith('i3d'):
        extractor_cls = features.pretrained.models.I3DExtractor
    elif model.startswith('clip_vitb32'):
        if 'pool' in model:
            extractor_cls = features.pretrained.models.ClipViTB32PooledExtractor
        else:
            extractor_cls = features.pretrained.models.ClipViTB32Extractor
    elif model.startswith('mvit_v1_b_16x2_random_stride32'):
        extractor_cls = features.pretrained.models.MViTV1B16x2Stride32RandomExtractor
    elif model.startswith('mvit_v1_b_16x2'):
        if 'stride32' in model:
            extractor_cls = features.pretrained.models.MViTV1B16x2Stride32Extractor
        else:
            extractor_cls = features.pretrained.models.MViTV1B16x2Extractor
    elif model.startswith('mvit_v1_b'):
        extractor_cls = features.pretrained.models.MViTV1BExtractor
    else:
        raise RuntimeError(f'Unrecognized model {model}')

    if return_class:
        return extractor_cls
    else:
        return extractor_cls.create(extractor_name=model, **kwargs)

def get_extractor_type(extractor) -> datasets.DatasetType:
    if isinstance(extractor, features.pretrained.models.VideoPretrainedModelExtractor):
        return datasets.DatasetType.CLIP
    else:
        return datasets.DatasetType.FRAME
