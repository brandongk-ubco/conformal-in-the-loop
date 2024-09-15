from torchvision.datasets import CelebA
import os
from torchvision.transforms import v2
import tqdm
import pandas as pd

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")

image_size = 224

resize = v2.Compose(
    [
        v2.Resize(image_size, max_size=image_size + 1),
        v2.CenterCrop(image_size),
    ]
)

for split in ["valid", "test"]:
    data = CelebA(
        os.environ.get("PATH_DATASETS", "./"),
        split=split,
        transform=resize,
        target_type="attr",
    )

    out_dir = os.path.join(PATH_DATASETS, "celeba", "processed", split)
    os.makedirs(out_dir, exist_ok=True)

    attrs = []

    idx = 0
    for example in tqdm.tqdm(data):
        example_attrs = dict(zip(list(data.attr_names), [int(i) for i in example[1]]))
        attrs.append(example_attrs)
        example[0].save(f"{out_dir}/{idx}.png")
        idx += 1

    attrs_df = pd.DataFrame(attrs)
    attrs_df.to_csv(f"{out_dir}/_attrs.csv", index=False)
