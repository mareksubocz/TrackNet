import json
import pandas as pd
from pathlib import Path

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
        
if __name__ == '__main__':
    root_path = '/Users/mareksubocz/Downloads/BeaverShort2/3mins-no-frames/'
    split_images_to_folders(root_path)
    join_jsons_to_csv(root_path)