import cv2
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from openai import OpenAI as OpenAIClient
import openai
import json
import os

class HueAnalyser:
    def __init__(self, openai_api_key: str = None):
        """
        Initialise the HueAnalyser with OpenCV and OpenAI API.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
        """
        # Set up OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    
    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Dict]:
        """
        Extract frames from video at specified intervals and return frame data with timestamps.
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Extract one frame every N frames
            
        Returns:
            List[Dict]: List of frames with timestamp and frame data
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                frames.append({
                    "timestamp": round(timestamp, 2),
                    "frame": frame
                })
            
            frame_count += 1
            
        cap.release()
        return frames
    
    def analyse_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyse a single frame to extract color information.
        
        Args:
            frame (np.ndarray): Frame data from OpenCV
            
        Returns:
            Dict: Analysis results including dominant colors and their percentages
        """
        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for hue channel
        hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
        hist = hist.flatten()
        
        # Normalize histogram
        hist = hist / hist.sum()
        
        # Find dominant colors (top 5)
        dominant_indices = np.argsort(hist)[-5:][::-1]
        dominant_colors = []
        
        for idx in dominant_indices:
            hue = idx
            percentage = float(hist[idx] * 100)
            
            # Convert hue to color name
            color_name = self.hue_to_color_name(hue)
            
            dominant_colors.append({
                "color": color_name,
                "hue": int(hue),
                "percentage": round(percentage, 2)
            })
        
        return {
            "dominant_colors": dominant_colors,
            "color_diversity": float(np.sum(hist > 0.01) / len(hist))  # Measure of color diversity
        }
    
    def hue_to_color_name(self, hue: int) -> str:
        """
        Convert hue value to color name.
        
        Args:
            hue (int): Hue value (0-179)
            
        Returns:
            str: Color name
        """
        # Define hue ranges for basic colors
        color_ranges = {
            (0, 10): "Red",
            (11, 20): "Orange-Red",
            (21, 30): "Orange",
            (31, 40): "Yellow-Orange",
            (41, 60): "Yellow",
            (61, 80): "Yellow-Green",
            (81, 100): "Green",
            (101, 120): "Cyan-Green",
            (121, 140): "Cyan",
            (141, 160): "Blue-Cyan",
            (161, 180): "Blue"
        }
        
        for (lower, upper), color in color_ranges.items():
            if lower <= hue <= upper:
                return color
        
        return "Unknown"
    
    def _get_theme_analysis(self, color_distribution: Dict[str, float]) -> Dict:
        """
        Use OpenAI API to analyse the color distribution and hypothesize about the video's theme.
        
        Args:
            color_distribution (Dict[str, float]): Distribution of colors in the video
            
        Returns:
            Dict: Theme analysis including mood, potential genre, and confidence score
        """
        prompt = f"""Based on the following color distribution in a video, analyze the potential theme, mood, and genre.
        Provide your analysis in JSON format with the following fields:
        - theme: main theme or subject matter
        - mood: emotional atmosphere
        - genre: potential video genre
        - score: 0-1 indicating confidence in the analysis
        - reasoning: brief explanation of your analysis

        Color Distribution:
        {json.dumps(color_distribution, indent=2)}

        Respond in JSON format only."""

        try:
            # Create a client instance with the API key
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Use the client with the new API structure
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a video analysis expert specialising in color theory and visual storytelling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return {
                "theme": "unknown",
                "mood": "unknown",
                "genre": "unknown",
                "confidence_score": 0,
                "reasoning": "Error in theme analysis"
            }
    
    def analyse_video(self, video_path: str) -> Tuple[Dict, List[Dict]]:
        """
        Analyse video file and return overall analysis and frame-by-frame breakdown.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Tuple[Dict, List[Dict]]: Overall analysis and frame analysis
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Analyse each frame
        frame_analysis = []
        color_distribution = {}
        total_frames = len(frames)
        
        for frame_data in frames:
            analysis = self.analyse_frame(frame_data["frame"])
            
            # Add to frame analysis
            frame_analysis.append({
                "timestamp": frame_data["timestamp"],
                "analysis": analysis
            })
            
            # Update color distribution
            for color_info in analysis["dominant_colors"]:
                color = color_info["color"]
                if color not in color_distribution:
                    color_distribution[color] = 0
                color_distribution[color] += 1
        
        # Calculate color distribution percentages
        color_distribution_percentages = {
            color: round(count/total_frames * 100, 2) 
            for color, count in color_distribution.items()
        }
        
        # Get theme analysis from OpenAI
        theme_analysis = self._get_theme_analysis(color_distribution_percentages)
        
        # Calculate overall analysis
        overall_analysis = {
            "primary_colors": sorted(
                [(color, count/total_frames) for color, count in color_distribution.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "color_distribution": color_distribution_percentages,
            "average_color_diversity": sum(frame["analysis"]["color_diversity"] 
                                         for frame in frame_analysis) / total_frames,
            "theme_analysis": theme_analysis,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_frames_analysed": total_frames
        }
        
        return overall_analysis, frame_analysis 