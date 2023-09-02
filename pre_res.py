# 预测图片类别
from ultralytics import YOLO


def read_txt(fname):
   with open(fname,'r',encoding='utf-8') as rp :
      return rp.readlines()
   

def lt_strip(lt,st=None):
    '''去除右侧字符(st)'''
    if st==None:
        return list(map(lambda x:x.rstrip(),lt))
    else :
        return list(map(lambda x:x.rstrip(st),lt))
    
def en_to_cn(res_cls,en_name,cn_name):
    if res_cls in en_name:
        return cn_name[en_name.index(res_cls)]
    else :return 'No Such Pets'

def sing_predictor(img,model_path):

    #加载模型进行预测
    model = YOLO(model_path)
    results = model.predict(img)

    #预测结果中置信度最高的类别的idx
    pre_idx = results[0].probs.top1
    #idx对应的class名 
    res_cls = results[0].names[pre_idx] 

    #加载数据集中的中英文名称
    cn_name = read_txt(r'all_cls_cn.txt')
    en_name = read_txt(r'all_cls_en.txt')

    #清洗标签
    cn_name = lt_strip(cn_name)
    en_name = lt_strip(en_name)
    
    return  en_to_cn(res_cls,en_name,cn_name)


if __name__ == '__main__':
    img = r'E:\py_code\pets-recognitions\test_pic\柯基.png'
    model_path = r'E:\py_code\pets-recognitions\runs\classify\train13\weights\best.pt'
    print(sing_predictor(img,model_path))


    
