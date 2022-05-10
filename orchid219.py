# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset class for Food-101 dataset."""

import csv
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import datasets
from datasets.tasks import ImageClassification


_BASE_URLS = {
    "orchid219": "https://drive.google.com/file/d/1oeFIMIbavcBbbHtZrNb1ga2DlSc8QzuC/view?usp=sharing",
}

_DL_URL = "{name}.zip"
_HOMEPAGE = ""
_DESCRIPTION = (
    "This dataset consists of orchid219 categories, with 101'000 images. For "
    "each class, 250 manually reviewed test images are provided as well as 750"
    " training images. On purpose, the training images were not cleaned, and "
    "thus still contain some amount of noise. This comes mostly in the form of"
    " intense colors and sometimes wrong labels. All images were rescaled to "
    "have a maximum side length of 512 pixels."
)

_CITATION = """\
 @inproceedings{bossard14,
  title = {orchid219 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
"""

_LICENSE = """\
LICENSE AGREEMENT
=================
 - The orchid219 data set consists of images from Foodspotting [1] which are not
   property of the Federal Institute of Technology Zurich (ETHZ). Any use beyond
   scientific fair use must be negociated with the respective picture owners
   according to the Foodspotting terms of use [2].

[1] http://www.foodspotting.com/
[2] http://www.foodspotting.com/terms/
"""

_NAMES = [i for i in range(219)]

class Orchid219(datasets.GeneratorBasedBuilder):
    """Orchid219 Images dataset."""
    DEFAULT_CONFIG_NAME = "orchid219"
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="orchid219", version=datasets.Version("1.0.0"), description="Orchid219 Image Classification"),
        datasets.BuilderConfig(name="public-test", version=datasets.Version("1.0.0"), description="Orchid219 Image Classification"),
    ]

    def _info(self):
        if self.config.name == 'public-test':
            features = datasets.Features(
                {
                    "filename":datasets.Value("string"),
                    "image": datasets.Image(),
                }
            )
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=features,  
                supervised_keys=None,
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
            
        features = datasets.Features(
            {
                "filename":datasets.Value("string"),
                "image": datasets.Image(),
                "category": datasets.ClassLabel(names=_NAMES),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  
            supervised_keys=("image", "category"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="category")],
        )

    def _get_drive_url(self, url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        from os.path import exists
        if self.config.name == 'public-test':
            root_PublicTestDatasets = '/content/drive/MyDrive/datasets/Orchid219/Public_Test'
            
            df = pd.DataFrame()
            df['file'] = [str(i) for i in list(Path(root_PublicTestDatasets).rglob("*.JPG"))]
            print(df.shape)
            df.to_csv(os.path.join(root_PublicTestDatasets, 'public-test.csv'), index=False)

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, 
                    gen_kwargs={
                        "archive_path": os.path.join(root_PublicTestDatasets, 'public-test.csv'),
                        'split':'test'
                        }
                    ),
            ]


        dl_path = os.path.join(os.getcwd(),'datasets',_DL_URL.format(name=self.config.name))
        if exists(dl_path):
            archive_path = dl_manager.download_and_extract(dl_path)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, 
                    gen_kwargs={
                        "archive_path": archive_path,
                        'split':'train'
                        }
                    ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, 
                    gen_kwargs={
                        "archive_path": archive_path,
                        'split':'validation'
                        }
                    ),
            ]
        else:
            archive_path = dl_manager.download_and_extract(self._get_drive_url(_BASE_URLS[self.config.name]))
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "archive_path": archive_path,
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "archive_path": archive_path,
                        "split": "validation",
                    },
                ),
                # datasets.SplitGenerator(
                #     name=datasets.Split.TEST,
                #     gen_kwargs={
                #         "filepath": archive_path,
                #         "split": "test",
                #     },
                # ),
            ]
    
    def _generate_examples(self, archive_path, split):
        if self.config.name == 'public-test':
            print(archive_path)
            df = pd.read_csv(archive_path, encoding="utf8")
            for uid,row in df.iterrows():
                filename = row
                image_file = filename
                yield uid, {"filename": image_file, "image":Image.open(image_file)}
        else:
            filepath = os.path.join(archive_path,self.config.name)
            csvPath = os.path.join(filepath,split+'.csv')  
            df = pd.read_csv(csvPath, encoding="utf8")
            for uid,row in df.iterrows():
                filename,category = row
                image_file = os.path.join(filepath,filename)
                yield uid, {"filename": image_file, "image":Image.open(image_file), "category": category}
                

