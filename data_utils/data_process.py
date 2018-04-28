import os
import re
import codecs


root = r'E:\datasets'
w = open('./data.txt', 'w')
labels = {"McDonald'sPic": 0, "McDonald'sWord": 1, "StarbucksPic": 2, "StarbucksWord": 3}
for file in os.listdir(root):
    if file[-3:] == 'txt':
        f = codecs.open(os.path.join(root, file), 'r', 'ansi')
        f = f.read()
        file_object = codecs.open(os.path.join(root, file), 'w', 'utf-8')
        file_object.write(f)
        f = open(os.path.join(root, file), 'r')
        for line in f:
            w.write(file[:-4]+' ')
            a = line.split('\t')
            file_root = a[0]
            file_name = file
            photo_detail = a[-1]
            object_detail = a[1:-1]
            w.write(str(len(object_detail))+' ')
            object_position = []
            for i in range(len(object_detail)):
                position = (re.search(pattern='face', string=object_detail[i]).span()[1]+3,
                            re.search(pattern='HardCase', string=object_detail[i]).span()[0] - 3)
                position = object_detail[i][position[0]:position[1]].split(',')  # 目标位置
                w.write(' '.join(position))
                w.write(' ')

                label = (re.search(pattern='Foodlabel', string=object_detail[i]).span()[1]+3,
                         re.search(pattern='FaceSize', string=object_detail[i]).span()[0] - 3)
                label = object_detail[i][label[0]:label[1]]
                w.write(str(labels[label])+' ')
            # print(object_detail, photo_detail)
        w.write('\n')
w.close()

# s = 'abcabc'
# pattern = 'a'
# a = re.search(pattern, s).span()
# print(re.findall(pattern, s))
# # print(s[a[1]+2:])
