import os
import json
from openai import OpenAI
from faster_whisper import WhisperModel as whisper
from moviepy.editor import VideoFileClip, AudioFileClip
from typing import List, Dict
from collections import Counter, defaultdict
from util import clean_and_merge_segments, extract_json_from_response

api_key = "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"

class AudioAnalysis:
    def __init__(self, model_name="base", gpt_model="gpt-4o-mini", output_folder='audio_results'):
        self.whisper_model= whisper(model_name)
        self.gpt_model = gpt_model
        self.output_folder = output_folder
        self.client = OpenAI(api_key=api_key)
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_audio(self, video_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(self.output_folder+"/audio.wav")
        print(f"🎧 Audio extracted to: {self.output_folder+"/audio.wav"}")
        return self.output_folder+"/audio.wav"

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
            chunk_clip.write_audiofile(self.output_folder+'/'+fname, codec='pcm_s16le')
            chunks.append((fname, start, end))
            i += 1
        return chunks

    def transcribe_chunks_with_timestamps(self, chunks):
        transcript_with_timestamps = []
        for fname, start, end in chunks:
           # with open(fname, "rb") as f:
            print(f"⏱️ Transcribing: {self.output_folder+'/'+fname}")
            resp = self.whisper_model.transcribe(self.output_folder+'/'+fname)
            transcript_with_timestamps.append({
                "start": start,
                "end": end,
                "text": resp["text"].strip()
            })
        return transcript_with_timestamps

    def transcribe_with_timestamps(self, audio_path: str):
        print(f"⏱️ Transcribing: {audio_path}...")
        segment_results, info = self.whisper_model.transcribe(audio_path)

        segments = [
            {
                "start":round(seg.start),
                "end": round(seg.end),
                "text": seg.text
            }
            for seg in segment_results
        ]
        print(f"⏱️ Done Transcribing: {audio_path}")
        return segments

    def repair_segments_with_gpt(self, segments, window_size=3):
        print(f"⏱️ Repairing segments...")
        repaired_results = []
        all_repaired_text = []

        for i in range(0, len(segments), window_size):
            print(f"⏱️ Repairing: segment {i}/{len(segments)}")
            
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

            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reconstructs and summarizes transcript segments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )

            output = response.choices[0].message.content

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

        print(f"⏱️ Done repairing text.")
        return repaired_results, "\n".join(all_repaired_text)

    def summarize_full_transcript(self, full_text):
        print(f"⏱️ Summarizing text...")
        prompt = (
            "Here is a complete reconstructed transcript of a video:\n\n"
            f"{full_text}\n\n"
            "Please summarize the entire content in 3–5 natural English sentences."
        )

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You summarize full transcripts clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        print(f"⏱️ Done summarizing text.")
        return response.choices[0].message.content.strip()

    def analyse_segments(self, segments: List[Dict], relevant_word:str) -> List[Dict]:
        print(f"⏱️ Analyzing segments...")
        analysed_segments = []
        for i in range(0, len(segments)):
            segment = segments[i]
            print(f"⏱️ Analyzing segment {i}/{len(segments)}: {segment['text']}")
        #for segment in segments:
            text = segment["text"]
            prompt = f"""
            Analyse the following dialogue segment and provide a JSON response with:
            1. dialogue_type (e.g., action, horror, comedy, romance, fantasy, drama, documentary, other)
            2. theme (main theme or topic of the dialogue)
            3. emotional_tone (primary emotional tone)
            4. score (0-1 indicating confidence in the analysis)
            5. key_points (list of main points discussed)
            6. relevant_score (0-1 indicating score of relevance with the target word)
            7. Summarize its main idea in 1–2 sentences.

            Target word: \"{relevant_word}\"
            Dialogue: \"{text}\"

            Respond in JSON format only, no markdowns. Format in the following JSON schema 
            {{
                "dialogue_type": text,
                "theme": text,
                "emotional_tone": text,
                "score": number,
                "key_points": array of text,
                "relevant_score": number,
                "main_idea": text
            }}
            """
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are a dialogue analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                cleaned = extract_json_from_response(response.choices[0].message.content)
                #print(cleaned)
                analysis = json.loads(cleaned)
            except Exception as e:
                print(f"Error in segment analysis: {e}")
                analysis = {
                    "dialogue_type": "unknown",
                    "theme": "unknown",
                    "emotional_tone": "unknown",
                    "score": 0,
                    "key_points": [],
                    "relevant_score": 0,
                    "main_idea": "unknown"
                }
            segment["analysis"] = analysis
            analysed_segments.append(segment)
        print(f"⏱️ Done analyzing segments.")
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
    
    def run_audio_analysis(self, video_path:str, output_path:str, relevant_word:str):      
        # Step 2: Transcribe + repair + analyze
        # ✅ 保存路径
        segments_path = os.path.join(output_path, 'audio_results', 'segments.json')

        # ✅ 如果 segments.json 存在，则直接加载；否则重新转录
        if os.path.exists(segments_path):
            print("🔁 Loading cached segments from JSON...")
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        else:
            # Step 1: Extract + split audio
            audio_path = self.extract_audio(video_path)
            #chunks = audio_analysis.split_audio_to_chunks(audio_path)

            print("📝 Transcribing audio...")
            segments = self.transcribe_with_timestamps(audio_path)
            with open(segments_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)

        # repaired, full_text = audio_analysis.repair_segments_with_gpt(segments)

        segments = clean_and_merge_segments(segments)

        full = []
        for seg in segments:
            full.append(seg['text'])
        full_text = "\n".join(full)
        
        analysed = self.analyse_segments(segments)
        summary = self.summarize_full_transcript(full_text)
        stats = self.summarize_analysis_distribution(analysed)

        audio_result = {
            "segments": analysed,
            "full_summary": summary,
            "distribution_summary": stats
        }

        with open(output_path+'/audio_analysis_results.json', "w", encoding="utf-8") as f:
            json.dump(audio_result, f, indent=2)
        print(f"✅ Audio analysis result saved to {output_path}")

        return audio_result

