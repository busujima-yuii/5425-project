def merge_segments(segments, min_seg_duration=2.0):
    merged = []
    buf = segments[0].copy()

    for seg in segments[1:]:
        seg_dur = seg["end"] - seg["start"]
        ends_with_punc = buf["text"].strip().endswith(('.', '?', '!'))
        if ends_with_punc and seg_dur >= min_seg_duration:
            merged.append(buf.copy())
            buf = seg.copy()
        else:
            buf["end"] = seg["end"]
            buf["text"] += seg["text"]
    merged.append(buf.copy())
    return merged
