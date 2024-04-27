

def cast_category_name_mapping_to_idx_mapping(class_name_mapping, source_class_name, target_class_name):
    """

    Args:
        class_name_mapping (_type_): _description_
        source_class_name (_type_): _description_
        target_class_name (_type_): _description_

    Returns:
        idx_mapping: {
            0: 0
            1: [2,3,4,5],
            2: 7
        }
    """
    idx_mapping = {}
    src_idx_counter = 0
    for tar_idx, name in enumerate(target_class_name):
        if name in class_name_mapping:
            src_name_list = class_name_mapping[name]
            src_idx_list = [source_class_name.index(src_name) for src_name in src_name_list]
            idx_mapping[tar_idx] = src_idx_list
            src_idx_counter += len(src_idx_list)
        else:
            idx_mapping[tar_idx] = src_idx_counter
            src_idx_counter += 1

    return idx_mapping
