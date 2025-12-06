# C'est Margiela — Fashion Image Retrieval Engine

C’est Margiela is a vision-based retrieval system that identifies iconic Maison Margiela garments from user-submitted photos.
Instead of classification, the system uses deep metric learning:

Embed images with SigLIP

Store all catalog embeddings in a FAISS index

Retrieve the nearest garments via k-NN search

This repository contains the core dataset manifests, embedding scripts, evaluation code, and the skeleton for an API that answers:

“Which Margiela piece is this?”
