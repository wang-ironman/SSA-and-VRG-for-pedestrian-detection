def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'ok')
        return True
    else:

        print(path + 'failed!')
        return False

def handle_result_for_eval(file_path, save_path):
    """
    :param file_path: "/home/user/PycharmProjects/handle_result/10_12/comp4_10_12_7_27_det_test_person.txt"
    :param save_path: "/home/user/PycharmProjects/handle_result/10_12/comp4_10_12_7_27_det_test_person/"
    :return: 
    """
    

    readfile = open(file_path)
    contx = readfile.readline()

    while contx:
        details = contx.replace("\n", "").split(" ")
        pathls = details[0].split('_')

        path_str = save_path
        mkdir(path_str +"/"+ pathls[0])
        wf = open(path_str +"/"+ pathls[0] +"/"+pathls[1]+".txt","a+")
        id = pathls[2][2:]
        id_int = int(id)
        if id_int - 1 != 0:
            wf.write(str(id_int + 1))
            wf.write(",")
            for i in range(2,6):
                if i < 4:
                    wf.write(str(details[i]))
                    wf.write(",")
                if i == 4:
                    wf.write(str(float(details[i])-float(details[2])))
                    wf.write(",")
                if i == 5:
                    print(details[i])
                    wf.write(str(float(details[i])-float(details[3])))
                    wf.write(",")

            wf.write(str(details[1]))
            wf.write("\n")
        contx = readfile.readline()

path_lst = [
    '/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/07_16_1.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/06_19_2.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/06_19_3.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/06_05_4.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/06_05_5.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/06_05_6.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-vis+seg/eval/2020_01_11_Sat_10_22_44.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-ad/eval/2020_01_07_Tue_11_36_13.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-ad/eval/2020_01_04_Sat_22_53_43.txt',
    #'/home/neuiva2/wangironman/SSD-change/pytorch-ssd-ad/eval/2020_01_03_Fri_16_00_37.txt',
]

for pth_idx in path_lst:
    save_path1 = pth_idx.split(".")[0]
    handle_result_for_eval(pth_idx, save_path1)


