#!/bin/bash
# 720p demo reel: third-person show clip with the robot head-camera view inset bottom-right.
# usage: make_demo2.sh out.mp4 "show.mp4|head.mp4|caption|speed|start_s|dur_s|head_offset_s" ...   (head may be "-": no inset)
# head_offset_s: head-clip time minus show-clip time; default = end-aligned (both recordings stop together).
# OUTRO="line 1;line 2;..." appends a closing card (first line large).
OUT=$1; shift
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
W=/tmp/demo2_parts; rm -rf $W; mkdir -p $W
esc() { echo "$1" | sed "s/'/\\'/g; s/:/\\:/g; s/,/\\,/g"; }
nice -n 19 ffmpeg -v error -y -f lavfi -i color=c=0x111418:s=1280x720:d=4:r=30 -vf "drawtext=fontfile=$F:text='FastSort':fontcolor=white:fontsize=96:x=(w-tw)/2:y=230,drawtext=fontfile=$F:text='Dual-arm, instruction-aware sorting for the Aalto G1 humanoid':fontcolor=0xcfd8dc:fontsize=32:x=(w-tw)/2:y=370,drawtext=fontfile=$F:text='Fully autonomous. Isaac Sim, scored episodes.':fontcolor=0x90a4ae:fontsize=24:x=(w-tw)/2:y=430" -c:v libx264 -pix_fmt yuv420p $W/p000.mp4
i=1
for spec in "$@"; do
  IFS="|" read -r s h cap sp st du ho <<< "$spec"; sp=${sp:-1}; st=${st:-0}
  dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
  if [ -n "$h" ] && [ "$h" != "-" ] && [ -z "$ho" ]; then ho=$(python3 -c "print(max(0.0, $(dur "$h") - $(dur "$s")))"); fi
  hst=$(python3 -c "print(max(0.0, $st + ${ho:-0}))")
  printf "%s" "$cap" > $W/cap$i.txt; D=""; [ -n "$du" ] && D="-t $du"
  BAR="drawbox=x=0:y=660:w=iw:h=60:color=black@0.6:t=fill,drawtext=fontfile=$F:textfile=$W/cap$i.txt:fontcolor=white:fontsize=28:x=24:y=676,drawtext=fontfile=$F:text='${sp}x':fontcolor=0xffd54f:fontsize=26:x=w-tw-20:y=16"
  if [ "$h" = "-" ] || [ -z "$h" ]; then
    nice -n 19 ffmpeg -v error -y -ss $st $D -i "$s" -vf "scale=1280:720,setpts=PTS/$sp,fps=30,$BAR" -an -c:v libx264 -pix_fmt yuv420p $W/p$(printf %03d $i).mp4
  else
    nice -n 19 ffmpeg -v error -y -ss $st $D -i "$s" -ss $hst $D -i "$h" -filter_complex "[0:v]scale=1280:720[b];[1:v]scale=384:288,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.8:t=3,drawtext=fontfile=$F:text='robot head camera':fontcolor=white:fontsize=18:x=8:y=8:box=1:boxcolor=black@0.5[i];[b][i]overlay=x=W-w-20:y=H-h-80:shortest=1,setpts=PTS/$sp,fps=30,$BAR" -an -c:v libx264 -pix_fmt yuv420p $W/p$(printf %03d $i).mp4
  fi
  i=$((i+1))
done
if [ -n "$OUTRO" ]; then
  VF="drawbox=x=0:y=0:w=iw:h=ih:color=0x111418:t=fill"; y=200; n=0
  IFS=";" read -ra LINES <<< "$OUTRO"
  for ln in "${LINES[@]}"; do
    printf "%s" "$ln" > $W/out$n.txt
    if [ $n = 0 ]; then fs=56; c=white; else fs=30; c=0xcfd8dc; fi
    VF="$VF,drawtext=fontfile=$F:textfile=$W/out$n.txt:fontcolor=$c:fontsize=$fs:x=(w-tw)/2:y=$y"
    y=$((y + (n == 0 ? 110 : 56))); n=$((n+1))
  done
  nice -n 19 ffmpeg -v error -y -f lavfi -i color=c=0x111418:s=1280x720:d=6:r=30 -vf "$VF" -c:v libx264 -pix_fmt yuv420p $W/p999.mp4
fi
ls $W/p*.mp4 | sed "s/^/file /" > $W/list.txt
nice -n 19 ffmpeg -v error -y -f concat -safe 0 -i $W/list.txt -c copy "$OUT" && ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT"
