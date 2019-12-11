with open('Z124117102_auto_clean.txt', encoding='latin-1') as t:
    lines = t.readlines()
    n = len(lines)
    line_set = []
    for i in range(1, n, 3):
        line_set.append(lines[i])

    with open('AfterTransfer.txt', 'w+', encoding='latin-1') as f:
        for line in line_set:
            f.writelines(line)

