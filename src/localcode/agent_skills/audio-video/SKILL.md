---
name: audio-video
description: Convert or compress audio and video files (m4a, mp3 input, wav, aiff, caf, mp4, mov). Use when the user asks to convert, shrink, re-encode or get information about a sound or video file.
---

Built into macOS, nothing to install.

Audio (`afconvert`, `afinfo`):
- To AAC .m4a: `afconvert -f m4af -d aac -b 128000 in.wav out.m4a`
- To WAV: `afconvert -f WAVE -d LEI16 in.m4a out.wav`
- To AIFF: `afconvert -f AIFF -d BEI16 in.mp3 out.aiff`
- Details: `afinfo in.m4a`
MP3 can be read but not written by macOS tools; offer .m4a, or ask before installing ffmpeg.

Video (`avconvert`):
- List presets: `avconvert --help`
- Re-encode / shrink: `avconvert --source in.mov --output out.mp4 --preset Preset1280x720`
- Audio only: `avconvert --source in.mov --output out.m4a --preset PresetAppleM4A`

Verify the output exists and `afinfo` / `mdls -name kMDItemDurationSeconds` reports a sensible duration, then tell the user the path.
