import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder first and then create a new one
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # Ensure random reproducibility
    random.seed(0)

    # Divide 30% of the data in the dataset into validation and test sets
    split_rate = 0.3

    # 指向data文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd)
    origin_path = os.path.join(data_root, 'data')
    assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

    data_class = [cla for cla in os.listdir(origin_path)
                    if os.path.isdir(os.path.join(origin_path, cla))]

    # Create a folder to save the training set
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in data_class:
        # Establish folders corresponding to each category
        mk_file(os.path.join(train_root, cla))

    # Create a folder to save the validation set
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in data_class:
        # Establish folders corresponding to each category
        mk_file(os.path.join(val_root, cla))

    # Create a folder to save the test set
    test_root = os.path.join(data_root, "test")
    mk_file(test_root)
    for cla in data_class:
        # Establish folders corresponding to each category
        mk_file(os.path.join(test_root, cla))

    for cla in data_class:
        cla_path = os.path.join(origin_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        pre_index = random.sample(images, k=int(num*split_rate))
        val_num = len(pre_index) // 2
        eval_index = list(pre_index[:val_num])
        test_index = list(pre_index[val_num:val_num * 2])
        for index, image in enumerate(images):
            if image in eval_index:

                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla, f'{cla}_{image}')
                copy(image_path, new_path)
            elif image in test_index:

                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla, f'{cla}_{image}')
                copy(image_path, new_path)
            else:

                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla, f'{cla}_{image}')
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()


