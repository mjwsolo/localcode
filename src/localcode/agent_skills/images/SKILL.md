---
name: images
description: Resize, convert, rotate, crop or compress images (png, jpeg, heic, gif, tiff, pdf pages). Use when the user asks to change an image's size, format or orientation, or to make thumbnails.
---

Built into macOS (`sips`), nothing to install. Write results to new files unless the user asks to overwrite.

- Resize so the longest side is N pixels: `sips -Z 1200 in.jpg --out out.jpg`
- Exact size: `sips -z <height> <width> in.png --out out.png`
- Convert format: `sips -s format jpeg in.heic --out out.jpg` (jpeg, png, heic, tiff, gif, bmp, pdf)
- JPEG quality: add `-s formatOptions 70`
- Rotate / flip: `sips -r 90 in.png --out out.png`, `sips -f horizontal ...`
- Crop around the centre: `sips -c <height> <width> in.png --out out.png`
- Size and format of an image: `sips -g pixelWidth -g pixelHeight -g format in.png`
- Many files: loop in the shell, e.g. `for f in *.png; do sips -Z 800 "$f" --out "thumbs/$f"; done`

Verify with `sips -g pixelWidth -g pixelHeight -g format` on the output, then tell the user the paths.
