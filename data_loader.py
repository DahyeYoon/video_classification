import glob
import os
from typing import Optional, Callable, Tuple, cast, Union, Dict, Any, List

from torch import Tensor
import torch
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset


class CUSTOM_DATASET(VisionDataset):

    """         
            └── custom_dataset/
                ├── video_data/
                │   ├── 1.avi
                │   └── ...
                ├── classInd.txt
                └── test_train_split/
                    ├── train.txt
                    └── test.txt

                    OR

            └── custom_dataset/
                ├── video_data/
                │   ├── class1/
                │   │   ├── class_1_a.avi 
                │   │   └── ...
                │   └── class2/
                │       ├── class_2_a.avi
                │       └── ...
                │   
                └── test_train_split/
                    ├── train.txt
                    └── test.txt
               
    """

    def __init__(
        self,
        dataset_name: str,
        root: str,
        annotation_path: str,
        classInd_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
    ) -> None:
        super().__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError(f"fold should be between 1 and 3, got {fold}")

        extensions = ("avi",)


        is_dir=os.path.isdir(os.path.join(self.root, os.listdir(self.root)[0]))
        if is_dir:
            self.classes, class_to_idx = find_classes(self.root)
            self.samples = make_dataset(
                self.root,
                class_to_idx,
                extensions,
            )
        else:
            self.classInd_path =classInd_path
            self.classes, class_to_idx = self._find_classes_from_clsfile(self.root, self.classInd_path)
            self.samples= self._make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_paths = [path for (path, _) in self.samples]
 
        video_clips = VideoClips(
            video_paths,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        # import torch
        # video_clips=torch.load('./video_clips.pth')

        self.dataset_name=dataset_name
        self.full_video_clips = video_clips
        self.train = train
        self.fold = fold
        self.indices = self._select_fold(video_paths, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform


    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    def _select_fold(self, video_list: List[str], annotation_path: str, fold: int, train: bool) -> List[int]:
        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, x.split("/")[-1]) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]

        return indices
    
    def _find_classes_from_clsfile(self, directory: str, cls_file: str) -> Tuple[List[str], Dict[str, int]]:
        with open(cls_file) as f:
            lines=f.readlines()
        lines=[x.split(" ") for x in lines]
        class_to_idx =  {cls[1].strip(): int(cls[0]) for cls in (lines)}
        classes=list(class_to_idx.keys())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        torch.save(class_to_idx, 'class_to_idx.pth')

        return classes, class_to_idx

    def _make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))
                # return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        classes_lower={}
        for cls in class_to_idx:
            classes_lower[cls.lower()]=cls

        instances = []
        available_classes = set()
        flist=sorted(os.listdir(directory))

        for file in flist:
            for target_class in classes_lower:
                if target_class in file.lower():
                    target_class=classes_lower[target_class]
                    class_index=class_to_idx[target_class]
                    path=os.path.join(directory, file)
                    if is_valid_file(path):
                        item= path, class_index
                        instances.append(item)
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances



    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, class_index
    
    def __get_dataset_name__(self):
        return self.dataset_name
