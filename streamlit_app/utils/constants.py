# constants.py

import os

# Get the project root directory (parent of streamlit_app)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Base paths relative to project root
DOC_ROOT_PATH = os.path.join(PROJECT_ROOT, 'Documents')
INDEX_ROOT_PATH = os.path.join(PROJECT_ROOT, 'Documents', 'index')  # For FAISS index
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'Documents', 'embeddings')  # For document embeddings

# Ensure directories exist
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
os.makedirs(INDEX_ROOT_PATH, exist_ok=True)

SUPPORTED_FILE_TYPES = ['txt', 'csv', 'docx', 'epub', 'hwp', 'ipynb', 
                        'jpeg', 'jpg', 'mbox', 'md', 'mp3', 
                        'mp4', 'pdf', 'png', 'ppt', 'pptm', 'pptx',
                        'xlsx', 'xls',
                        ]
