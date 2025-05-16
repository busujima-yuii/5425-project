## Setup Guide

1. Manually install ffmpeg on local machine and check if the installation is successful:
   ```bash
   ffmpeg -version
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Replace openai_api_key variable with key

## Usage for Audio Analyser

1. Place your video file in the same directory
2. Inside main.py, find video_path, and replace the directory of your video with "video_placeholder"
3. Run the main script:
   ```bash
   python main.py
   ```

## Output

The analyser generates a detailed analysis of the video content, saved in the `results` directory as `{video_name}_results.json`. The analysis includes:

### Overall Analysis
- Primary Dialogue Type (e.g., action, horror, comedy, romance, fantasy)
- Primary Theme
- Primary Emotional Tone
- Key Points from the video

### Segment Analysis
For each segment of the video, the analysis includes:
- Start and end timestamps
- Transcribed text
- Dialogue type
- Theme
- Emotional tone
- Confidence score
- Key points for that segment

### Example Output Structure
```json
{
  "overall_analysis": {
    "primary_dialogue_type": "comedy",
    "dialogue_type_distribution": {
      "comedy": 3,
      "drama": 2
    },
    "primary_theme": "family",
    "theme_distribution": {
      "family": 2,
      "friendship": 1
    },
    "primary_emotional_tone": "happy",
    "emotional_tone_distribution": {
      "happy": 3,
      "excited": 2
    },
    "key_points": [
      "Family activities",
      "Friendship moments",
      "Learning experiences"
    ],
    "analysis_timestamp": "2024-03-14T15:30:45.123456",
    "total_segments": 5
  },
  "segment_analysis": [
    {
      "start": 0,
      "end": 5,
      "text": "Transcribed text from the video",
      "analysis": {
        "dialogue_type": "comedy",
        "theme": "family",
        "emotional_tone": "happy",
        "confidence_score": 0.92,
        "key_points": [
          "Point 1",
          "Point 2"
        ]
      }
    }
  ]
}
```

## Notes

- The analysis quality depends on the audio quality of the input video
- Processing time varies based on video length and complexity
