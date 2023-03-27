import os
import sys

if __name__ == '__main__':
    print('argv count:', len(sys.argv))
    print('argv str:', str(sys.argv))

    input_path_dir = sys.argv[1]
    online_path_dir = sys.argv[2]

    oss_dir = f'oss://ai-audio-test/{online_path_dir}'
    for input_file in input_path_dir:
        if input_file.startswith("."):
            continue

        file_path = os.path.join(input_path_dir, input_file)
        oss_path = os.path.join(oss_dir, input_file)

        command = '../ossutil64 cp {} {}'.format(file_path, oss_path)
        print(f'upload file:{input_file}, online path:{oss_path}')
        os.system(command)