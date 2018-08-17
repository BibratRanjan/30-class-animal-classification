import pandas as pd
import os, shutil
import cv2
import glob

def make_dir():
    df = pd.read_csv('./DL_Beginner/meta-data/train.csv')
    col2 = df['Animal'].tolist()
    
    classes = list(set(col2))
    #print(len(classes))
    
    """Creating appropriate folders"""
    original_dataset_dir = './DL_Beginner/train'
    base_dir = './Dataset'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    val_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    
    for single_class in classes:
        class_dir_train = os.path.join(train_dir, single_class)
        if not os.path.exists(class_dir_train):
            os.mkdir(class_dir_train)
            
        class_dir_val = os.path.join(val_dir, single_class)
        if not os.path.exists(class_dir_val):
            os.mkdir(class_dir_val)
            
    """Transferring data to train folder and then to validation folder"""
    
    file_name = "Faulty_images_names"
    text_file = open(file_name + ".txt", "w")
    
    for index in range(13000):
        img_id = df.iloc[index, 0]
        animal = df.iloc[index, 1]
        
        src_addr = os.path.join(original_dataset_dir, img_id)
        dest_addr = os.path.join(train_dir, animal, img_id)
        img = cv2.imread(src_addr)
        if img is None:
            text_file.write(animal + '/' + img_id + "\n")
        else:
            shutil.copyfile(src_addr, dest_addr)
    
    #Sanity check
    class_dict = {}
    global_count = 0
    for single_class in classes:
        class_dir_train = os.path.join(train_dir, single_class)
        count = len(glob.glob(class_dir_train + '/*.jpg'))
        class_dict[single_class] = count
        global_count = global_count + count
    print(class_dict)
    print(global_count)
        
    rename_files(train_dir, classes)
    data_augmentation(train_dir, classes)
    validation_partition(train_dir, classes)
    
    #Sanity check for training
    class_dict = {}
    global_count = 0
    for single_class in classes:
        class_dir_train = os.path.join(train_dir, single_class)
        count = len(glob.glob(class_dir_train + '/*.jpg'))
        class_dict[single_class] = count
        global_count = global_count + count
    print(class_dict)
    print(global_count)
    
    #Sanity check for validation
    class_dict = {}
    global_count = 0
    for single_class in classes:
        class_dir_train = os.path.join(val_dir, single_class)
        count = len(glob.glob(class_dir_val + '/*.jpg'))
        class_dict[single_class] = count
        global_count = global_count + count
    print(class_dict)
    print(global_count)
    
def validation_partition(train_dir, classes):
    parts = train_dir.split(os.path.sep)
    for single_class in classes:
        src_dir = os.path.join(parts[0], parts[1], parts[2], single_class)
        dest_dir = os.path.join(parts[0], parts[1], 'validation', single_class)
        for i in range(1, 241):
            src = os.path.join(src_dir, str(i).zfill(4) + '.jpg')
            dest = os.path.join(dest_dir, str(i).zfill(4) + '.jpg')
            shutil.move(src, dest)

def rename_files(train_dir, classes):
    for single_class in classes:
        class_dir_train = os.path.join(train_dir, single_class)
        images = glob.glob(class_dir_train + '/*.jpg')
        
        count = 1
        for image in images:
            parts = image.split(os.path.sep)
            src = os.path.join(parts[0], parts[1], parts[2], parts[3], parts[4])
            dest = os.path.join(parts[0], parts[1], parts[2], parts[3], str(count).zfill(4) + ".jpg")
            os.rename(src, dest)
            count += 1

def data_augmentation(train_dir, classes):
    
    req_frames = 1200
    for single_class in classes:
        class_dir_train = os.path.join(train_dir, single_class)
        generated_files = sorted(glob.glob(class_dir_train + '/*.jpg'))
    
        nb_additional_frames = req_frames - len(generated_files)
        nb_outer_loop = nb_additional_frames // len(generated_files)
        residue = nb_additional_frames % len(generated_files)
    
        count = len(generated_files) + 1
        for _ in range(nb_outer_loop):
            for j in range(1, len(generated_files) + 1):
                shutil.copy(os.path.join(class_dir_train, str(j).zfill(4) +'.jpg'), 
                            os.path.join(class_dir_train, str(count).zfill(4) +'.jpg'))
                count += 1
    
        for i in range(1, residue + 1):
            shutil.copy(os.path.join(class_dir_train, str(i).zfill(4) +'.jpg'), 
                            os.path.join(class_dir_train, str(count).zfill(4) +'.jpg'))
            count += 1
    
def main():
    make_dir()
    
if __name__ == '__main__':
    main()
        
        
    
    
    
    
    



