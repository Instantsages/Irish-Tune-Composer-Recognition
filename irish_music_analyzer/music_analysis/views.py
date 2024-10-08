from .utils import processing_pipeline
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from .forms import TuneForm
from .models import Tune
import json

def index(request):
    return HttpResponse("Hello, welcome to the Irish Music Analyzer!")

def music_dashboard(request):
    return render(request, 'dashboard.html')  # Make sure to create this template

def discover(request):
    return render(request, 'discover.html')

def tunes(request):
    # List all tunes (Read)
    tunes = Tune.objects.all()

    # Check if a tune is being updated
    tune_id = request.GET.get('edit')  # Get 'edit' parameter from the query string
    delete_id = request.GET.get('delete')  # Get 'delete' parameter from the query string
    form = None

    # Debugging: Check if the request is a POST
    if request.method == 'POST':
        print("POST request received")

    # Handle Create/Update form
    if request.method == 'POST':
        if tune_id:
            # Update tune
            tune = get_object_or_404(Tune, pk=tune_id)
            form = TuneForm(request.POST, instance=tune)
        else:
            # Create new tune
            form = TuneForm(request.POST)

        if form.is_valid():
            form.save()
            return redirect('tunes')

    elif tune_id:
        # Populate form for editing
        tune = get_object_or_404(Tune, pk=tune_id)
        form = TuneForm(instance=tune)
    
    if delete_id and request.method == 'POST':
        # Debugging: Check if we are inside the delete logic
        print(f"Trying to delete tune with ID: {delete_id}")
        tune = get_object_or_404(Tune, pk=delete_id)
        tune.delete()
        print(f"Tune deleted: {delete_id}")
        return redirect('tunes')

    return render(request, 'tunes.html', {
        'tunes': tunes,
        'form': form,
        'tune_id': tune_id,
        'delete_id': delete_id
    })

@csrf_exempt
def get_musical_features_data(request):
    if request.method == 'POST':
        # Parse the JSON body to get selected X, Y, and Z features
        body = json.loads(request.body)
        x_feature = body.get('xFeature')
        y_feature = body.get('yFeature')
        z_feature = body.get('zFeature')

        # Fetch all tunes
        tunes = Tune.objects.all()

        # Get all abc_notations
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Pass to pipeline to run the KMeans algorithm
        tunes_extracted_features = processing_pipeline(abc_notations)

        # Composer Color Mapping
        composer_color_mapping = {
            'Sean Ryan': 'red',
            'Paddy Fahey': 'yellow',
            'Lizz Carrol': 'green'
        }

        # Initialize lists to hold the extracted features for X, Y, and Z axes
        x_data = []
        y_data = []
        z_data = []
        labels = []
        colors = []

        # Get selected features for each tune
        for tune_name, features in tunes_extracted_features.items():
            labels.append(tune_name)
            x_data.append(features.get(x_feature))
            y_data.append(features.get(y_feature))
            z_data.append(features.get(z_feature))
            colors.append(composer_color_mapping.get(features.get('composer')))

        # Prepare the response data
        return JsonResponse({
            'x': x_data,        # X-axis data
            'y': y_data,        # Y-axis data
            'z': z_data,        # Z-axis data
            'labels': labels,   # Tune names
            'composerColorMapping': colors
        })
