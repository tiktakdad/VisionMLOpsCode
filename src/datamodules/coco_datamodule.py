import os
from pathlib import Path
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import src.utils.transforms as T

class COCODataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.coco_dataset_path = Path(self.hparams.data_dir)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # like getter
    @property
    def num_classes(self) -> int:
        # number of coco's classes
        return 80

    def prepare_data(self) -> None:
        """Download data if needed."""
        if os.path.exists(self.coco_dataset_path) is not True:
            # Download labels
            from src.utils.general import download
            segments = False  # segment or box labels
            #url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
            #urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
            #download(urls, dir=self.coco_dataset_path.parent)
            # Download data
            urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
                    'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
                    'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
            download(urls, dir=self.coco_dataset_path, threads=3)
            urls = ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                    'http://images.cocodataset.org/annotations/image_info_test2017.zip']
            download(urls, dir=self.coco_dataset_path, threads=2)

    def make_coco_transforms(self, image_set, fix_size=False, strong_aug=False, args=None):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # config the params for data aug
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        scales2_resize = [400, 500, 600]
        scales2_crop = [384, 600]

        # # update args from config files
        # scales = getattr(args, 'data_aug_scales', scales)
        # max_size = getattr(args, 'data_aug_max_size', max_size)
        # scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
        # scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

        # # resize them
        # data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
        # if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        #     data_aug_scale_overlap = float(data_aug_scale_overlap)
        #     scales = [int(i*data_aug_scale_overlap) for i in scales]
        #     max_size = int(max_size*data_aug_scale_overlap)
        #     scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        #     scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

        # datadict_for_print = {
        #     'scales': scales,
        #     'max_size': max_size,
        #     'scales2_resize': scales2_resize,
        #     'scales2_crop': scales2_crop
        # }
        # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

        if image_set == 'train':
            if fix_size:
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ])

            if strong_aug:
                from src.utils import sltransform as SLT

                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose([
                            T.RandomResize(scales2_resize),
                            T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ])
                    ),
                    SLT.RandomSelectMulti([
                        SLT.RandomCrop(),
                        # SLT.Rotate(10),
                        SLT.LightingNoise(),
                        SLT.AdjustBrightness(2),
                        SLT.AdjustContrast(2),
                    ]),
                    # # for debug only
                    # SLT.RandomCrop(),
                    # SLT.LightingNoise(),
                    # SLT.AdjustBrightness(2),
                    # SLT.AdjustContrast(2),
                    # SLT.Rotate(10),
                    normalize,
                ])

            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
            ])

        if image_set in ['val', 'test']:

            if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
                print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
                return T.Compose([
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ])

            return T.Compose([
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ])

        raise ValueError(f'unknown {image_set}')

    def build(self, image_set):
        mode = 'instances'
        PATHS = {
            "train": (self.coco_dataset_path / "train2017", self.coco_dataset_path / "annotations" / f'{mode}_train2017.json'),
            "train_reg": (self.coco_dataset_path / "train2017", self.coco_dataset_path / "annotations" / f'{mode}_train2017.json'),
            "val": (self.coco_dataset_path / "val2017", self.coco_dataset_path / "annotations" / f'{mode}_val2017.json'),
            "eval_debug": (self.coco_dataset_path / "val2017", self.coco_dataset_path / "annotations" / f'{mode}_val2017.json'),
            "test": (self.coco_dataset_path / "test2017", self.coco_dataset_path / "annotations" / 'image_info_test-dev2017.json'),
        }

        # add some hooks to datasets
        aux_target_hacks_list = None
        img_folder, ann_file = PATHS[image_set]
        # load datasets only if they're not loaded already
        dataset = CocoDetection(img_folder, ann_file,
                                transforms=self.make_coco_transforms(image_set))
        return dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.build('train')
            self.data_val = self.build('val')
            self.data_test = self.build('test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )