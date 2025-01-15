from glob import glob
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def enrich_data(sdir="train"):
    imgs = glob(os.path.join(DATASET_FOLDER, sdir, "*", "*.jpg"))
    img_folders = [p.split(os.path.sep) for p in imgs]
    img_names = [p[2] for p in img_folders]
    labels = [p[1] for p in img_folders]
    labels = list(map(float, labels))
    labels_log = np.log(labels)
    mm = MinMaxScaler()
    mm_log_labels = mm.fit_transform(labels_log.reshape(-1, 1))
    mm_log_labels = mm_log_labels.ravel()

    split = [p[0] for p in img_folders]

    # dataframe = pd.DataFrame({'img_path': imgs, 'img_names': img_names, 'labels': labels, 'split': split})
    dataframe = pd.DataFrame({'img_path': imgs, 'img_names': img_names, 'labels': mm_log_labels, 'split': split})
    return dataframe


if __name__ == "__main__":
    DATASET_FOLDER = "./"
    train_data = enrich_data()
    val_data = enrich_data('val')
    test_data = enrich_data('test')
    merge = [train_data, val_data, test_data]
    merge_data = pd.concat(merge)
    # merge_data = pd.merge(train_data, val_data, how='outer')
    print(merge_data)
    merge_data.to_csv('./data.csv', index=False)
