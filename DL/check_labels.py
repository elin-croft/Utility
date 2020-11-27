import os, sys


root = sys.argv[1]
data = []
classes = set()
classes_file = {}
if __name__ == "__main__":
    files = os.listdir(root)
    for xml_file in files:
        if xml_file.endswith('.xml'):
            with open(os.path.join(root, xml_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if '<name>' in line:
                        classes.add(line[line.find('<name>') + 6:line.find('</')])
                        classes_file.setdefault(line[8:-8], set()).add(xml_file)
    print(classes)
    for k, v in classes_file.items():
        print(k + ': ', len(v))
        classes_file[k] = len(v)
    # labels = open('labels.txt', 'w')
    # labels.write(str(classes) + '\n')
    # labels.write(str(classes_file) + '\n')
    # labels.close()