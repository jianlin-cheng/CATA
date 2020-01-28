
import numpy as np

np.random.seed(0)

def load_tags_data(data_dir, num_items):
    data = {}
    data["tags"] = load_file(data_dir+"tag-item.dat")
    data["citations"] = load_file(data_dir+"citations.dat")
    
    num_tags= len(data["tags"])
    print ("Number of tags: ", num_tags)
    item_tag_matrix = np.zeros(shape=(num_items,num_tags))
    for i in range(num_tags):
        item_ids = data["tags"][i]
        for j in item_ids:
            item_tag_matrix[j,i] = 1
    
    item_tag_matrix_with_citations = item_tag_matrix.copy()
    for i in range(len(data["citations"])):
        item_ids = data["citations"][i]
        for j in item_ids:
            for k in range(num_tags):
                if (item_tag_matrix[j][k] == 1):
                    item_tag_matrix_with_citations[i][k] = item_tag_matrix[j][k]

    return item_tag_matrix_with_citations

def load_file(path):
    arr = []
    for line in open(path):
        a = line.strip().split()
        if a[0]==0:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr

