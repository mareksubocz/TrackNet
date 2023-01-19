import json
import pandas as pd
from pathlib import Path
import sys

def split_images_to_folders(root_path):
    for f in Path(root_path).glob('img/*'):
        movie_path = Path(str(f)[:str(f).find('.mp4')+4])
        movie_path.mkdir(exist_ok=True)
        frame_number = int(str(f)[str(f).rfind('_')+1:str(f).rfind('.jpg')])
        # moves to folder and renames to frame number
        f.rename( (movie_path / str(frame_number)).with_suffix(f.suffix) )


def join_jsons_to_csv(root_path, relative=True):
    root_path = Path(root_path)
    for video in root_path.glob('img/*'):
        if not video.is_dir():
            continue
        print(video)

        res = {'num': [], 'x': [], 'y': [], 'visible': [], 'height': [], 'width': []}
        for metadata in root_path.glob(f'ann/{video.name}*.json'):
            with open(metadata, 'r') as f:
                data = json.load(f)
                video_width = data['size']['width']
                video_height = data['size']['height']
                for obj in data['objects']:
                    res['num'].append(int(str(metadata)[str(metadata).rfind('_')+1:str(metadata).rfind('.jpg')]))
                    x1 = obj['points']['exterior'][0][0]
                    x2 = obj['points']['exterior'][1][0]
                    y1 = obj['points']['exterior'][0][1]
                    y2 = obj['points']['exterior'][1][1]
                    res['x'].append((x1+x2)/2)
                    res['y'].append((y1+y2)/2)
                    res['visible'].append(1)
                    res['height'].append(obj['points']['exterior'][1][1] - obj['points']['exterior'][0][1])
                    res['width'].append(obj['points']['exterior'][1][0] - obj['points']['exterior'][0][0])

                    # read only the first object
                    break

        df = pd.DataFrame.from_dict(res).sort_values(by=['num'])
        df['x'] /= video_width
        df['y'] /= video_height
        df['width'] /= video_width
        df['height'] /= video_height
        df.to_csv(root_path / (video.name+'.csv'), index=False)
        

#TODO: read video tags to root_path/video_tags.csv
def jsons_to_csv(root_path):
    root_path = Path(root_path)
    file_tags = {'filename': [], 'view': [], 'ball_type': []}
    for filename in (root_path).glob('*.json'):
        filename_cleaned = Path(Path(filename.stem).stem)
        actions_start = []
        actions_end = []
        res = {'num': [], 'x': [], 'y': [], 'visible': [], 'height': [], 'width': [], 'inPlay': []}
        file_tags['filename'].append(filename_cleaned)
        with open(filename, 'r') as f:
            root = json.load(f)
            video_width = root['size']['width']
            video_height = root['size']['height']
            for tag in root['tags']:
                if tag['name'] == 'View':
                    file_tags['view'].append(tag['value'])
                elif tag['name'] == 'BallType':
                    file_tags['ball_type'].append(tag['value'])
                elif tag['name'] == 'ActionStartEnd':
                    if tag['value'] == 'start':
                        actions_start.append(int(tag['frameRange'][0]))
                    else:
                        actions_end.append(int(tag['frameRange'][0]))
            actions_start = sorted(actions_start)
            actions_end = sorted(actions_end)
            print(filename)
            print(actions_start)
            print(actions_end)
            if actions_end[0] < actions_start[0]:
                actions_end = actions_end[1:]
                
            for frame in root['frames']:
                res['num'].append(frame['index'])
                x1 = frame['figures'][0]['geometry']['points']['exterior'][0][0]
                y1 = frame['figures'][0]['geometry']['points']['exterior'][0][1]
                x2 = frame['figures'][0]['geometry']['points']['exterior'][1][0]
                y2 = frame['figures'][0]['geometry']['points']['exterior'][1][1]
                res['x'].append(((x1+x2)/2) / video_width)
                res['y'].append(((y1+y2)/2) / video_height)
                res['visible'].append(1)
                res['width'].append((x2-x1) / video_width)
                res['height'].append((y2-y1) / video_height)
                for action_range in zip(actions_start, actions_end):
                    if int(frame['index']) >= action_range[0] and int(frame['index']) < action_range[1]:
                        res['inPlay'].append(1)
                        break
                else:
                    res['inPlay'].append(0)
        
        df = pd.DataFrame.from_dict(res).sort_values(by=['num'])
        print(df)
        df.to_csv(root_path / (filename_cleaned.with_suffix('.csv')), index=False)
        df_tags = pd.DataFrame.from_dict(file_tags)
        df_tags.to_csv(root_path / 'video_tags.csv', index=False)

if __name__ == '__main__':
    jsons_to_csv(sys.argv[1])