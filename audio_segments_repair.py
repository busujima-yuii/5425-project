import openai
import json

openai.api_key = "YOUR_OPENAI_API_KEY"

def repair_segments_with_gpt(segments, window_size=3):
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reconstructs and summarizes transcript segments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )

        output = response.choices[0].message["content"]
        
        # 🔍 Extract info using pattern (or GPT-assumed structure)
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

def summarize_full_transcript(full_text):
    prompt = (
        "Here is a complete reconstructed transcript of a video:\n\n"
        f"{full_text}\n\n"
        "Please summarize the entire content in 3–5 natural English sentences."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You summarize full transcripts clearly and concisely."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message["content"].strip()