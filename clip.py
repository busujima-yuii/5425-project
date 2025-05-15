import ffmpeg
from tempfile import NamedTemporaryFile

def clip(path, start_time, end_time):
    video_bytes = None
    with open(path, "rb") as f:
        video_bytes = f.read()

    with NamedTemporaryFile(suffix=".mp4", delete=False) as in_tmp:
        in_tmp.write(video_bytes)
        in_path = in_tmp.name

    out_tmp = NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tmp.name
    out_tmp.close()

    (
        ffmpeg
        .input(in_path, ss=start_time, to=end_time)
        .output(out_path, c='copy')
        .run(quiet=True, overwrite_output=True)
    )

    with open(out_path, "rb") as f:
        clipped = f.read()
    return clipped
