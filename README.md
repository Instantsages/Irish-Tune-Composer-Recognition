# Irish-Tune-Composer-Recognition

This project is a web-based platform designed to analyze Irish music tunes. Users can visualize musical features in an interactive 3D scatter plot. The application enables users to select specific musical features for visualization and customize their graphs with options such as axis selections.

## Current Features

### 1. **Tune Management**
- Add new tunes with their ABC notation.
- View, update, and delete existing tunes.

### 2. **Musical Feature Extraction**
- Extract musical features such as:
  - **Notes**: Number of tunes.
  - **Rests**: The length of the tune in minutes.
  - **Chords**: Number of chords.
  - **etc.**

### 3. **3D Data Visualization**
- Visualize musical features on a 3D scatter plot.
- Select different features to be plotted on the X, Y, and Z axes.
- Hover over points to see the tune names for each data point.

## Technologies Used

- **Backend**: Django (Python)
- **Frontend**: HTML, CSS (with responsive design), JavaScript (Plotly.js for interactive plots)
- **Database**: SQLite (default Django database for development)
- **Styling**: Custom CSS for a modern UI

## Installation and Setup

### Prerequisites
- **Python 3.6+** installed on your machine.
- **pip** for managing Python packages.
- **virtualenv** (optional but recommended) for isolating dependencies.

### 1. Clone the repository
```bash
git clone https://github.com/Instantsages/Irish-Tune-Composer-Recognition.git
source myenv/bin/activate
cd irish_music_analyzer
python manage.py runserver 
