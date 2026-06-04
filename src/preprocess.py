import os
import cv2
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def resize_and_save(src_path, dst_path, img_size=(64,64)):
    img = cv2.imread(str(src_path))

    if img is None:
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    cv2.imwrite(
        str(dst_path),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

    return True

def build_dataset(raw_dir, out_dir, img_size=(64,64), test_size=0.1, val_size=0.1):

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(
        [d.name for d in raw_dir.iterdir() if d.is_dir()]
    )

    label_map = {cls:i for i,cls in enumerate(classes)}

    with open(out_dir / "labels.json","w") as f:
        json.dump(label_map,f,indent=2)

    files = []
    labels = []

    for cls in classes:
        for img_path in (raw_dir/cls).glob("*"):
            files.append(str(img_path))
            labels.append(cls)

    f_train, f_temp, l_train, l_temp = train_test_split(
        files,
        labels,
        test_size=test_size+val_size,
        stratify=labels,
        random_state=42
    )

    rel_val = val_size/(test_size+val_size)

    f_val, f_test, l_val, l_test = train_test_split(
        f_temp,
        l_temp,
        test_size=rel_val,
        stratify=l_temp,
        random_state=42
    )

    def write_split(file_list,label_list,split_name):

        for src,lbl in tqdm(
            zip(file_list,label_list),
            total=len(file_list),
            desc=f"Saving {split_name}"
        ):

            dst_dir = out_dir / split_name / lbl

            dst_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            dst = dst_dir / Path(src).name

            resize_and_save(
                src,
                dst,
                img_size
            )

    write_split(f_train,l_train,"train")
    write_split(f_val,l_val,"val")
    write_split(f_test,l_test,"test")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--size", type=int, default=64)

    args = parser.parse_args()

    build_dataset(
        args.raw,
        args.out,
        img_size=(args.size,args.size)
    )
