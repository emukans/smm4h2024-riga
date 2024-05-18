import csv
from contextlib import suppress


class Meddra(object):
    '''Basic Meddra Entity object'''
    def __init__(self, ptid, lltid, text):
        self.ptid = ptid
        self.lltid = lltid
        self.text = text


def get_meddra_dict(meddra_llt):
    """load corpus data and write resolution files"""
    pt_dict, llt_dict = {}, {}
    for line in open(meddra_llt, 'r'):
        elems = line.split("$")
        if len(elems) > 2:
            ptid, lltid, text = elems[2], elems[0], elems[1]
            entry = Meddra(ptid, lltid, text)
            if ptid == lltid:
                pt_dict[ptid] = entry
            llt_dict[lltid] = entry
    return pt_dict, llt_dict


def downcast_dev_set():
    meddra_path = '../data/task1/Resource/llt.asc'
    gt_data_path = '../data/task1/Dev_2024/gold_annotations_complete.tsv'
    gt_downcasted_data_path = '../data/task1/Dev_2024/norms_downcast.tsv'
    seen_downcasted_data_path = '../data/task1/Dev_2024/seen_downcast.txt'
    pt_dict, llt_dict = get_meddra_dict(meddra_path)

    downcasted_result = []
    seen_list = []
    with open(gt_data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            tag_id = line[-1]
            while True:
                if llt_dict[tag_id].ptid == tag_id:
                    seen_list.append(tag_id)
                    break
                tag_id = llt_dict[tag_id].ptid

            downcasted_result.append('\t'.join(line[:-1] + [tag_id]))

    with open(gt_downcasted_data_path, 'w') as f:
        f.write('\n'.join(set(downcasted_result)) + '\n')

    with open(seen_downcasted_data_path, 'w') as f:
        f.write('\n'.join(seen_list) + '\n')


def downcast_train_set():
    meddra_path = '../data/task1/Resource/llt.asc'
    gt_data_path = '../data/task1/Train_2024/gold_annotations_complete.tsv'
    gt_downcasted_data_path = '../data/task1/Train_2024/train_spans_norm_downcast.tsv'
    seen_downcasted_data_path = '../data/task1/Train_2024/train_seen_downcast.txt'
    pt_dict, llt_dict = get_meddra_dict(meddra_path)

    downcasted_result = []
    seen_list = []

    with open(gt_data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            tag_id = line[-1]
            while True:
                if llt_dict[tag_id].ptid == tag_id:
                    break
                tag_id = llt_dict[tag_id].ptid

            seen_list.append(tag_id)
            downcasted_result.append('\t'.join(line[:-1] + [tag_id]))

    with open(gt_downcasted_data_path, 'w') as f:
        f.write('\n'.join(set(downcasted_result)) + '\n')

    with open(seen_downcasted_data_path, 'w') as f:
        f.write('\n'.join(seen_list) + '\n')


if __name__ == '__main__':
    downcast_dev_set()
    downcast_train_set()
