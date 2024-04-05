#!/bin/bash
set -eux

INPUT_BASELINE="$1"
INPUT_OURS="$2"
OUTPUT="$3"

#: "${VIDEO_MODE:=HALF}"
: "${VIDEO_MODE:=SWEEP}"

: "${DRAW_TEXT:=ON}"
: "${DRAW_BAR:=ON}"
: "${CROP_TO_HD_ASPECT:=OFF}"

if [ $CROP_TO_HD_ASPECT == "ON" ]; then
    BASE_FILTER="
        [0:v]crop=iw:'min(ih,iw/16*9)'[base];\
        [1:v]crop=iw:'min(ih,iw/16*9)'[ours]"
else
    BASE_FILTER="[0:v]copy[base];[1:v]copy[ours]"
fi
if [ $DRAW_TEXT == "ON" ]; then
    BASE_FILTER="
        $BASE_FILTER;\
        [base]drawtext=text='Splatfacto':fontcolor=white:fontsize=h/50:x=w/50:y=h/50[base];\
        [ours]drawtext=text='DN-Splatter':fontcolor=white:fontsize=h/50:x=w-tw-w/50:y=h/50[ours]"
fi
if [ $DRAW_BAR == "ON" ]; then
    BASE_FILTER="
        $BASE_FILTER;\
        color=0x80ff80,format=rgba[bar];\
        [bar][base]scale2ref[bar][base];\
        [bar]crop=iw:ih/200:0:0[bar];\
        [ours][bar]overlay=x=0:y=0[ours]"
fi

case $VIDEO_MODE in
  HALF)
    VIDEO_FILTER="
        $BASE_FILTER;\
        [base]crop=iw/2:ih:0:0[left_crop];\
        [ours]crop=iw/2:ih:iw/2:0[right_crop];\
        [left_crop][right_crop]hstack"
    ;;

  SWEEP)
    LEN=8
    VIDEO_FILTER="
        $BASE_FILTER;\
        color=0x00000000,format=rgba,scale=[black];\
        color=0xffffffff,format=rgba[white];\
        [black][base]scale2ref[black][base];\
        [white][base]scale2ref[white][base];\
        [white][black]blend=all_expr='if(lte(X,W*abs(1-mod(T,$LEN)/$LEN*2)),B,A)'[mask];\
        [ours][mask]alphamerge[overlayalpha]; \
        [base][overlayalpha]overlay=shortest=1"
    ;;

  *)
    echo -n "unknown video mode $VIDEO_MODE"
    exit 1
    ;;
esac

ffmpeg -i "$INPUT_BASELINE" -i "$INPUT_OURS" -filter_complex "$VIDEO_FILTER" -hide_banner -y "$OUTPUT"