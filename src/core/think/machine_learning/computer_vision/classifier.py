# src/core/think/machine_learning/computer_vision/classifier.py

import numpy as np
import pickle
from src.core.machine import Machine
from src.utils import plot
from sklearn.metrics.pairwise import cosine_similarity
import cv2

class MatchingClassifier(Machine):
    """
    MatchingClassifier machine that classifies face embeddings to identify individuals.
    """

    def __init__(self, machine_name, state_manager):
        super().__init__(machine_name, state_manager)
        config = self.config.get('config', {})
        self.confidence_threshold = config.get('confidence_threshold', 0.80)
        embeddings_file = config.get('embeddings_file', './models/re_train_data_face.pkl')
        self.train_embeddings, self.train_labels = self.load_embeddings_and_labels(embeddings_file)

    def load_embeddings_and_labels(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        embeddings = np.array(data['embeddings'])
        labels = data['labels']
        if self.logger:
            self.logger.info(f"Loaded embeddings and labels from {filename}")
        return embeddings, labels

    def process(self, input_data):
        """
        Matches embeddings against known embeddings to classify faces.
        """
        embeddings = input_data.get('embeddings')
        bboxes = input_data.get('bboxes', [])
        frame = input_data.get('frame')

        labels = []
        confidences = []
        if embeddings is None or len(embeddings) == 0:
            if self.logger:
                self.logger.warning("No embeddings to process.")
            return {
                "labels": labels,
                "confidences": confidences,
                "frame": frame
            }

        embeddings = np.squeeze(np.array(embeddings), axis=1)  # Ensure correct shape

        for bbox, embedding in zip(bboxes, embeddings):
            predicted_label, score = self.match_faces(embedding)
            if score < self.confidence_threshold:
                predicted_label = "Not Recognized"

            labels.append(predicted_label)
            confidences.append(score)

            # Optionally, plot the results on the frame
            frame = plot.plot_faces_with_labels(frame, bbox, predicted_label, score)

        processed_data = {
            "labels": labels,
            "confidences": confidences,
            "frame": frame
        }

        if self.logger:
            self.logger.info(f"Classified {len(labels)} faces.")

        return processed_data

    def match_faces(self, test_embedding):
        # Compute cosine similarity
        similarities = cosine_similarity(test_embedding.reshape(1, -1), self.train_embeddings)
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[0][best_match_index]
        predicted_label = self.train_labels[best_match_index]

        return predicted_label, best_match_score

    def send_data(self, processed_data):
        """
        Sends classification results to the shared data_dict.
        """
        super().send_data(processed_data)

    def on_start(self):
        super().on_start()
        if self.logger:
            self.logger.info("MatchingClassifier machine started.")

    def on_finish(self):
        super().on_finish()
        if self.logger:
            self.logger.info("MatchingClassifier machine finished.")
