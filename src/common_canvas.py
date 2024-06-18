import time
from typing import Any, Callable, List

import numpy as np
from streaming import Stream, StreamingDataset
from src.streaming_dataset import StreamingDataset
from PIL import Image
import io
from tqdm import tqdm

class CommonCanvasDataset(StreamingDataset):
    def __init__(self,
                 filepath: List[str],
                 shuffle: bool = False,
                 batch_size: int = None,
                 transform: Callable = lambda x: x
                ) -> None:
        start = time.time()
        if isinstance(filepath, str):
            filepath = [fp for fp in filepath.split(";")]
        streams = [Stream(local=fp) for fp in tqdm(filepath, "preparing streams")]
        super().__init__(streams=streams, shuffle=shuffle, batch_size=batch_size, sampling_method="fixed")
        self.transform = transform
        end = time.time()
        print("Dataset initialized took", end-start, "seconds")

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        img = Image.open(io.BytesIO(obj["jpg"]))
        cpt = obj["blip2_caption"]
        return self.transform(img), cpt


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    filename = "../../CC/0/;../../CC/01/"
    ds = CommonCanvasDataset(filepath=filename)#, transforms=transforms(size=256, crop_size=256))
    for (img, cpt) in ds:
        print(np.array(img).shape, cpt)
    img, cpt = ds[0]
