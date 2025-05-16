import whisper
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import numpy as np
import openai
import json
import os
from datetime import datetime

# 本来写的现在不用了
# Define genre categories and their associated keywords with weights
GENRE_KEYWORDS = {
    'Action': {
        'fight': 5, 'battle': 5, 'war': 5, 'attack': 4, 'defend': 4,
        'weapon': 4, 'explosion': 4, 'chase': 4, 'hero': 3, 'villain': 3,
        'danger': 3, 'mission': 3, 'rescue': 3, 'combat': 3, 'action': 3
    },
    'Horror': {
        'scary': 5, 'fear': 5, 'horror': 5, 'terrifying': 5, 'monster': 4,
        'ghost': 4, 'haunted': 4, 'dark': 3, 'creepy': 4, 'scream': 3,
        'death': 3, 'blood': 3, 'evil': 3, 'nightmare': 3, 'kill': 3
    },
    'Comedy': {
        'funny': 5, 'laugh': 5, 'joke': 4, 'humor': 4, 'comedy': 4,
        'hilarious': 4, 'silly': 3, 'fun': 3, 'amusing': 3, 'entertaining': 3,
        'witty': 3, 'humorous': 3, 'comic': 3, 'playful': 2, 'joy': 2
    },
    'Romance': {
        'love': 5, 'romance': 5, 'heart': 4, 'kiss': 4, 'relationship': 4,
        'passion': 4, 'affection': 3, 'dating': 3, 'couple': 3, 'romantic': 3,
        'soulmate': 3, 'marriage': 3, 'wedding': 3, 'sweet': 2, 'tender': 2
    },
    'Fantasy': {
        'magic': 5, 'fantasy': 5, 'magical': 4, 'wizard': 4, 'dragon': 4,
        'enchanted': 4, 'spell': 3, 'mythical': 3, 'legend': 3, 'myth': 3,
        'supernatural': 3, 'fairy': 3, 'kingdom': 3, 'quest': 3, 'adventure': 3
    }
}

# Common words to ignore (stop words)
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

class AudioSummariser:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the AudioSummariser with Whisper model and OpenAI API.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
        """
        self.model = whisper.load_model("base")
        
        # Set up OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio file and return segments with timestamps.
        
        Args:
            audio_path (str): Path to the audio/video file
            
        Returns:
            List[Dict]: List of segments with start time, end time, and text
        """
        result = self.model.transcribe(audio_path)
        return [
            {
                "start": round(seg["start"]),
                "end": round(seg["end"]),
                "text": seg["text"]
            }
            for seg in result["segments"]
        ]
    
    def analyse_segment(self, text: str) -> Dict:
        """
        Analyse a text segment using OpenAI's API to identify dialogue type, theme, and emotional content.
        
        Args:
            text (str): Text segment to analyse
            
        Returns:
            Dict: Analysis results including dialogue type, theme, and emotional content
        """
        prompt = f"""Analyse the following dialogue segment and provide a JSON response with:
        1. dialogue_type (one of: action, horror, comedy, romance, fantasy, drama, documentary, other)
        2. theme (main theme or topic of the dialogue)
        3. emotional_tone (primary emotional tone)
        4. confidence_score (0-1 indicating confidence in the analysis)
        5. key_points (list of main points discussed)

        Dialogue: "{text}"

        Respond in JSON format only."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a dialogue analysis expert. Provide concise, accurate analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return {
                "dialogue_type": "unknown",
                "theme": "unknown",
                "emotional_tone": "unknown",
                "confidence_score": 0,
                "key_points": []
            }
    
    def summarise_audio(self, audio_path: str) -> Tuple[Dict, List[Dict]]:
        """
        Analyse audio file and return overall analysis and segment-by-segment breakdown.
        
        Args:
            audio_path (str): Path to the audio/video file
            
        Returns:
            Tuple[Dict, List[Dict]]: Overall analysis and segment analysis
        """
        # Transcribe audio
        segments = self.transcribe_audio(audio_path)
        
        # Analyse each segment
        segment_analysis = []
        dialogue_type_counts = defaultdict(int)
        themes = defaultdict(int)
        emotional_tones = defaultdict(int)
        all_key_points = []
        
        for segment in segments:
            analysis = self.analyse_segment(segment["text"])
            
            # Add to segment analysis
            segment_analysis.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "analysis": analysis
            })
            
            # Update overall statistics
            dialogue_type_counts[analysis["dialogue_type"]] += 1
            themes[analysis["theme"]] += 1
            emotional_tones[analysis["emotional_tone"]] += 1
            all_key_points.extend(analysis["key_points"])
        
        # Calculate overall analysis
        total_segments = len(segments)
        overall_analysis = {
            "primary_dialogue_type": max(dialogue_type_counts.items(), key=lambda x: x[1])[0],
            "dialogue_type_distribution": dict(dialogue_type_counts),
            "primary_theme": max(themes.items(), key=lambda x: x[1])[0],
            "theme_distribution": dict(themes),
            "primary_emotional_tone": max(emotional_tones.items(), key=lambda x: x[1])[0],
            "emotional_tone_distribution": dict(emotional_tones),
            "key_points": list(set(all_key_points)),  # Remove duplicates
            "analysis_timestamp": datetime.now().isoformat(),
            "total_segments": total_segments
        }
        
        return overall_analysis, segment_analysis


