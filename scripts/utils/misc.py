def get_scene_id(input_file_name, sc_len):
    return int(input_file_name.split('/')[-1].split('_')[0][:sc_len])


def get_track_id(input_file_name, sc_len):
    return int(input_file_name.split('/')[-1].split('_')[0][sc_len:])


def get_cam_id(input_file_name):
    return int(input_file_name.split('/')[-1].split('_')[1].split('s')[0].split('c')[1])


def get_source_id(input_file_name, vs_len):
    return int(input_file_name.split('/')[-1].split('_')[2][:vs_len])


def get_view(input_file_name):
    return int(input_file_name.split('/')[-1].split('_')[3].split('.')[0])
