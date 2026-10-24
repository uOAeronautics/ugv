"""Combine the reference video with rendered integer-second room views.

python3 make_room_comparison.py REFERENCE.mp4 --frames-dir preview/room/every-second
"""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('--no-anchors', action='store_true', help='Omit compass labels for other reference videos')
parser.add_argument('--frames-dir', type=Path, default=Path(__file__).parent/'preview/room/every-second')
args = parser.parse_args()
folder = args.frames_dir.resolve()
def run(*command):
    subprocess.run(command, check=True)
meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(args.reference)]))
duration = float(meta['format']['duration'])
count = int(duration)+1
for i in range(count):
    if not (folder/f'{i:02d}.png').is_file():
        parser.error(f'Missing rendered view {i:02d}.png')
refs = folder/'reference'
pairs = folder/'pairs'
overlays = folder/'labels'
refs.mkdir(parents=True, exist_ok=True)
pairs.mkdir(parents=True, exist_ok=True)
overlays.mkdir(parents=True, exist_ok=True)
common = ['ffmpeg','-hide_banner','-loglevel','error','-y']
run(*common,'-i',str(args.reference),'-vf',"select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)'",'-fps_mode','vfr','-start_number','0',str(refs/'%02d.png'))
font = Path('/System/Library/Fonts/Helvetica.ttc')
face = ImageFont.truetype(str(font),22) if font.exists() else ImageFont.load_default(size=22)
sheet = Image.new('RGB',(1920,1368),'#24272a')
anchors={4:'EAST / LONG SIDE',8:'SOUTH',11:'WINDOW SIDE / WEST',17:'BACK / NORTH'}
if args.no_anchors: anchors={}
for i in range(count):
    labels = Image.new('RGBA',(1152,1024),(0,0,0,0))
    draw = ImageDraw.Draw(labels)
    timestamp=f'{i:02d}s'+(' | '+anchors[i] if i in anchors else '')
    timestamp_x=int((1152-draw.textlength(timestamp,font=face))/2)
    for x,y,text in ((12,12,'REFERENCE'),(588,12,'MODEL - ESTIMATED CAMERA'),(timestamp_x,975,timestamp)):
        box=draw.textbbox((x,y),text,font=face)
        draw.rectangle((box[0]-6,box[1]-5,box[2]+6,box[3]+5),fill=(0,0,0,180))
        draw.text((x,y),text,font=face,fill='white')
    labels.save(overlays/f'{i:02d}.png')
    pair=Image.new('RGBA',(1152,1024))
    pair.paste(Image.open(refs/f'{i:02d}.png').convert('RGBA').resize((576,1024)),(0,0))
    pair.paste(Image.open(folder/f'{i:02d}.png').convert('RGBA').resize((576,1024)),(576,0))
    pair=Image.alpha_composite(pair,labels).convert('RGB')
    pair.save(pairs/f'{i:02d}.png')
    sheet.paste(pair.resize((384,342),Image.Resampling.LANCZOS),((i%5)*384,(i//5)*342))
sheet.save(folder/'all-seconds.png')
filters='[0:v]scale=576:1024,setsar=1[left];[1:v]scale=576:1024,setsar=1[right];[left][right]hstack=inputs=2[pair];[pair][2:v]overlay=eof_action=repeat[out]'
run(*common,'-i',str(args.reference),'-framerate','1','-start_number','0','-i',str(folder/'%02d.png'),
    '-framerate','1','-start_number','0','-i',str(overlays/'%02d.png'),
    '-filter_complex',filters,'-map','[out]','-t',str(duration),'-r','25',
    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'room-video-comparison.mp4'))
print(folder/'room-video-comparison.mp4')
