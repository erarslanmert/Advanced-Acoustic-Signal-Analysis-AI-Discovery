import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

# We assume yamnet_class_map.csv is local after download
class_map_path = 'yamnet_class_map.csv'

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with open(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def run_test():
    try:
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("Model loaded successfully")
        
        class_names = class_names_from_csv(class_map_path)
        print(f"Loaded {len(class_names)} class names")
        
        # input is 1D waveform, at 16000 hz
        waveform = np.zeros(32000, dtype=np.float32) # 2 seconds of silence
        scores, embeddings, spectrogram = model(waveform)
        
        print("Scores shape:", scores.shape) # (n_frames, 521)
        mean_scores = np.mean(scores, axis=0)
        top_n = np.argsort(mean_scores)[::-1][:3]
        
        print("Top 3 classes on silence:")
        for i in top_n:
            print(f"  {class_names[i]}: {mean_scores[i]:.3f}")
            
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    run_test()
