import os
from copy import deepcopy
import librosa
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from PIL import Image
import soundfile as sf
import av
from decord import VideoReader
from decord import cpu
from torch.utils.data import DataLoader
import torchaudio
from transformers import Wav2Vec2Processor
from model.audio_model.wav2vec import Wav2Vec2

torchaudio.set_audio_backend("sox_io")


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img


class ReactionDatasetDepthShort(data.Dataset):
    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751,
                 fps=25, load_video=True,
                 load_3dmm=True, load_ref=True):

        self._root_path = root_path
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self._data_path = os.path.join(self._root_path, self._split)
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        self._list_path = self._list_path.drop(0)

        self.load_video = load_video
        self.load_3dmm = load_3dmm

        self.dataset_path = os.path.join(root_path, self._split)
        # self._video_path = os.path.join(self.dataset_path, 'Video_files')
        self._video_path = os.path.join(self.dataset_path, 'Video')
        self._3dmm_path = os.path.join(self.dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = [path for path in list(self._list_path.values[:, 1])]
        listener_path = [path for path in list(self._list_path.values[:, 2])]
        speaker_path_tmp = speaker_path + listener_path
        listener_path_tmp = listener_path + speaker_path
        speaker_path = speaker_path_tmp
        listener_path = listener_path_tmp

        self.speaker_path = speaker_path.copy()
        self.listener_path = listener_path.copy()

        self.data_list = [path for path in list(self._list_path.values[:, 1])] + [path for path in list(
            self._list_path.values[:, 2])]  # the data_list is actually the same as speaker_path


        self._len = len(self.data_list)  # 3186

    def __getitem__(self, index):
        total_length = 751
        cp = random.randint(150, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        # ========================= Load Speaker & Listener video clip ==========================

        # ====== load speaker video clip
        video_clip = torch.zeros(size=(0,))
        if self.load_video:
            video_path = os.path.join(self._video_path, self.speaker_path[index] + '.mp4')
            clip = []
            with open(video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp, cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img)

            # shape: [_clip_length, 3, 224, 224]
            video_clip = torch.stack(clip, dim=0)

        # ========================= Load Listener 3DMM ==========================
        _3dmm_clip = torch.zeros(size=(0,))
        # ====== load listener 3dmm
        if self.load_3dmm:
            _3dmm_path = os.path.join(self._3dmm_path, self.speaker_path[index] + '.npy')
            _3dmm = torch.FloatTensor(np.load(_3dmm_path)).squeeze()
            _3dmm = _3dmm[cp: cp + self._clip_length]
            _3dmm_clip = self._transform_3dmm(_3dmm)[0]


        return (
            video_clip,
            _3dmm_clip,
        )

    def __len__(self):
        return self._len


def get_dataloader_depth_short(conf):
    assert conf.split in ["train", "val", "test"], "split must be in [train, val, test]"
    print('==> Preparing data for {}...'.format(conf.split) + '\n')

    dataset = ReactionDatasetDepthShort(
        root_path=conf.dataset_path,
        split=conf.split,
        img_size=conf.img_size,
        crop_size=conf.crop_size,
        clip_length=conf.clip_length,
        fps=conf.fps,
        load_video=conf.load_video,
        load_3dmm=conf.load_3dmm,
    )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=conf.batch_size,
                            shuffle=conf.shuffle,
                            num_workers=conf.num_workers)
    return dataloader
