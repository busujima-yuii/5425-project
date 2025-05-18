import os
import json
import openai
from moviepy import VideoFileClip, AudioFileClip
from typing import List, Dict
from collections import Counter, defaultdict

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual key

class AudioAnalysisPipeline:
    def __init__(self, model_name="whisper-1", gpt_model="gpt-4"):
        self.whisper_model = model_name
        self.gpt_model = gpt_model

    def extract_audio(self, video_path, output_path="audio.wav"):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path)
        print(f"🎧 Audio extracted to: {output_path}")
        return output_path

    def split_audio_to_chunks(self, audio_path, chunk_length=30):
        """
        Split audio into chunks using moviepy instead of pydub.
        
        Args:
            audio_path: Path to input audio .wav
            chunk_length: Length in seconds (default = 30)
        
        Returns:
            List of (filename, start_time, end_time)
        """
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        chunks = []
        i = 0
        while i * chunk_length < duration:
            start = i * chunk_length
            end = min((i + 1) * chunk_length, duration)
            chunk_clip = audio.subclip(start, end)
            fname = f"chunk_{int(start)}_{int(end)}.wav"
            chunk_clip.write_audiofile(fname, codec='pcm_s16le')
            chunks.append((fname, start, end))
            i += 1
        return chunks

    def transcribe_chunks_with_timestamps(self, chunks):
        transcript_with_timestamps = []
        for fname, start, end in chunks:
            with open(fname, "rb") as f:
                print(f"⏱️ Transcribing: {fname}")
                resp = openai.Audio.transcribe(self.whisper_model, f)
                transcript_with_timestamps.append({
                    "start": start,
                    "end": end,
                    "text": resp["text"].strip()
                })
        return transcript_with_timestamps

    def repair_segments_with_gpt(self, segments, window_size=3):
        repaired_results = []
        all_repaired_text = []

        for i in range(0, len(segments), window_size):
            group = segments[i:i+window_size]
            if not group:
                continue

            start = group[0]["start"]
            end = group[-1]["end"]

            combined_text = "\n".join([f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}" for seg in group])

            prompt = (
                "The following transcript segments were generated from audio chunks. "
                "Each may be incomplete or contain broken sentences due to time slicing.\n\n"
                f"{combined_text}\n\n"
                "Please rewrite the merged transcript into a natural, coherent paragraph. "
                "Then summarize its main idea in 1–2 sentences.\n\n"
                "Output format:\n"
                "Repaired Start Time: <number>\n"
                "Repaired End Time: <number>\n"
                "Repaired Transcript: <reconstructed full paragraph>\n"
                "Summary: <short summary>"
            )

            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reconstructs and summarizes transcript segments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )

            output = response.choices[0].message["content"]

            def extract_section(label):
                try:
                    return output.split(label + ":")[1].split("\n", 1)[0].strip()
                except:
                    return ""

            new_start = float(extract_section("Repaired Start Time") or start)
            new_end = float(extract_section("Repaired End Time") or end)
            new_text = extract_section("Repaired Transcript")
            summary = extract_section("Summary")

            repaired_results.append({
                "start": new_start,
                "end": new_end,
                "transcript": new_text,
                "summary": summary
            })

            all_repaired_text.append(f"[{new_start:.1f}-{new_end:.1f}] {new_text}")

        return repaired_results, "\n".join(all_repaired_text)

    def summarize_full_transcript(self, full_text):
        prompt = (
            "Here is a complete reconstructed transcript of a video:\n\n"
            f"{full_text}\n\n"
            "Please summarize the entire content in 3–5 natural English sentences."
        )

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You summarize full transcripts clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        return response.choices[0].message["content"].strip()

    def analyse_repaired_segments(self, repaired_segments: List[Dict]) -> List[Dict]:
        analysed_segments = []
        for segment in repaired_segments:
            text = segment["transcript"]
            prompt = f"""
            Analyse the following dialogue segment and provide a JSON response with:
            1. dialogue_type (e.g., action, horror, comedy, romance, fantasy, drama, documentary, other)
            2. theme (main theme or topic of the dialogue)
            3. emotional_tone (primary emotional tone)
            4. score (0-1 indicating confidence in the analysis)
            5. key_points (list of main points discussed)

            Dialogue: \"{text}\"

            Respond in JSON format only.
            """
            try:
                response = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are a dialogue analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                analysis = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error in segment analysis: {e}")
                analysis = {
                    "dialogue_type": "unknown",
                    "theme": "unknown",
                    "emotional_tone": "unknown",
                    "score": 0,
                    "key_points": []
                }
            segment["analysis"] = analysis
            analysed_segments.append(segment)
        return analysed_segments

    def summarize_analysis_distribution(self, analysed_segments: List[Dict]) -> Dict:
        dialogue_type_counter = Counter()
        theme_counter = Counter()
        emotion_counter = Counter()
        scores = []
        all_keypoints = set()

        for seg in analysed_segments:
            a = seg.get("analysis", {})
            dialogue_type_counter[a.get("dialogue_type", "unknown")] += 1
            theme_counter[a.get("theme", "unknown")] += 1
            emotion_counter[a.get("emotional_tone", "unknown")] += 1
            scores.append(a.get("score", 0))
            all_keypoints.update(a.get("key_points", []))

        total = len(analysed_segments)
        return {
            "total_segments": total,
            "primary_dialogue_type": dialogue_type_counter.most_common(1)[0][0],
            "dialogue_type_distribution": dict(dialogue_type_counter),
            "primary_theme": theme_counter.most_common(1)[0][0],
            "theme_distribution": dict(theme_counter),
            "primary_emotional_tone": emotion_counter.most_common(1)[0][0],
            "emotional_tone_distribution": dict(emotion_counter),
            "average_score": round(sum(scores)/total, 3) if total > 0 else 0,
            "key_points": sorted(all_keypoints)
        }
