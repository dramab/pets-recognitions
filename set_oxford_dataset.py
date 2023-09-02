import os 
import shutil 
import random

# 指定存储数据集的位置
dataset = 'dataset'

# 源数据集的位置
place = r"E:\py_code\大创\Pets-Face-Recognition-main\pets_datasets\oxford-iiit-pet\images"
path = os.listdir(place)

#删除.mat结尾的文件索引
for i in path:
    if '.mat' in i :
        path.remove(i)
len(path)

#删除名称中多余的数字与后缀
path1 = list(map(lambda x : x.rstrip('_0123456789.jpg'),path))


#获取索引并将索引分类
class_id = {}
for idx in range(len(path)):
    
    file_name = path1[idx]
    if file_name not in class_id:
        class_id[file_name] = [path[idx]]
    else:
        class_id[file_name].append(path[idx])

#将数据集分为train和val两类
train_dic = {}
val_dic = {}
for cls in class_id:
    val_dic[cls] = random.sample(class_id[cls],5)
    train_dic[cls] = list(set(class_id[cls])-set(val_dic[cls]))

train_path = os.path.join(dataset,'train')
val_path = os.path.join(dataset,'val')
os.makedirs(train_path,exist_ok = True)
os.makedirs(val_path,exist_ok = True)

def set_folder(folder_dic,name):
    for folder_pic in folder_dic:
        #生成测试集
        os.makedirs(os.path.join(dataset,name,folder_pic))
        for pic in folder_dic[folder_pic]:
            shutil.copy(os.path.join(place,pic),os.path.join(dataset,name,folder_pic))

set_folder(train_dic,'train')
set_folder(val_dic,'val')