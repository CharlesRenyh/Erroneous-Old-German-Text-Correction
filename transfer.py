import os

ori_path = '/Users/charlesren/Downloads/AI_lab_data/Data/'
for book in os.listdir(ori_path):
    with open(ori_path + book, 'r', encoding='latin-1') as t:
        lines = t.readlines()
        n = len(lines)
        line_set = []
        for i in range(1, n, 3):
            line_set.append(lines[i])

        with open('AfterTransfer.txt', 'w+') as f:
            for line in line_set:
                f.writelines(line)

