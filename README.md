# Anxiety Support Chatbot

Anxiety Support Chatbot is an AI-powered mental health assistant designed to provide empathetic, evidence-informed support for people experiencing different types of anxiety. [memory:3][memory:12] It combines emotion detection, intent and scenario routing, and RAG-based therapeutic content to deliver context-aware, conversational guidance rather than one-off answers. [memory:4][memory:6][memory:11]

## Features

- Emotion detection using a DistilBERT-based classifier trained on annotated data to infer the user’s emotional state from text.   
- Intent and scenario routing that maps messages into specific anxiety scenarios (e.g., panic, health, uncertainty, social, isolation) and triggers tailored flows.  
- Retrieval-Augmented Generation (RAG) that pulls CBT-style psychoeducation, grounding techniques, and coping strategies from a curated therapeutic knowledge base using a vector store.   
- Clinical flows engine that runs structured, multi-step conversations (7+ flows) with interruption logic so the bot can switch scenarios when the user’s state changes.
- Personalization layer that logs interaction data and user preferences to adapt tone, encouragement, and technique selection over time. 
- Web frontend (React) with a simple chat UI that displays messages along with model intent/emotion scores for debugging and experimentation. [

## Tech Stack

- Backend: Python, FastAPI, modular pipeline for preprocessing, emotion detection, intent classification, scenario routing, and clinical flow management.   
- NLP & ML: PyTorch, DistilBERT-based emotion classifier, custom intent and scenario models. 
- Retrieval: ChromaDB as the vector database for therapeutic content embeddings and semantic search.  
- Frontend: React-based chat interface with REST integration to the FastAPI backend.

## Project Goals

- Provide accessible, always-available anxiety support that complements, not replaces, professional care. 
- Implement a research-informed, modular architecture that can be extended with safety layers, human-in-the-loop review, and additional therapeutic flows over time. 

> Note: This project is for educational and supportive purposes only and is **not** a substitute for professional diagnosis, treatment, or emergency care.
