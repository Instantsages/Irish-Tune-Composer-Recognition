from .utils import processing_pipeline
from django.shortcuts import render, get_object_or_404, redirect
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from sklearn.cluster import KMeans
from itertools import combinations
from .forms import TuneForm
from .models import Tune
import numpy as np
import pandas as pd
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
    
    tunes_data = {}
    for tune in tunes:
        tunes_data[tune.pk] = {
            'name': tune.name,
            'composer': tune.composer,
            'abc_notation': tune.abc_notation,
        }

    return render(request, 'tunes.html', {
        'tunes': tunes,
        'form': form,
        'tune_id': tune_id,
        'tunes_data_json': json.dumps(tunes_data),
        'delete_id': delete_id
    })

def tunes_add(request):
    if request.method == 'POST':
        form = TuneForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({'success': True})
        else:
            # Render the form with errors
            html = render_to_string('tunes_form_partial.html', {'form': form}, request=request)
            return JsonResponse({'success': False, 'html': html})
    else:
        form = TuneForm()
    return render(request, 'tunes_form_partial.html', {'form': form})

def tunes_edit(request, pk):
    tune = get_object_or_404(Tune, pk=pk)
    if request.method == 'POST':
        form = TuneForm(request.POST, instance=tune)
        if form.is_valid():
            form.save()
            return JsonResponse({'success': True})
        else:
            # Render the form with errors
            html = render_to_string('tunes_form_partial.html', {'form': form, 'tune': tune}, request=request)
            return JsonResponse({'success': False, 'html': html})
    else:
        form = TuneForm(instance=tune)
    return render(request, 'tunes_form_partial.html', {'form': form, 'tune': tune})

def tunes_delete(request, pk):
    print("Trying to delete tune")
    tune = get_object_or_404(Tune, pk=pk)
    if request.method == 'POST':
        tune.delete()
        return JsonResponse({'success': True, 'pk': pk})
    return render(request, 'tunes_confirm_delete_partial.html', {'tune': tune})

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
    
def perform_clustering(request):
    if request.method == 'POST':
        # Parse the JSON body to get selected X, Y, and Z features
        body = json.loads(request.body)
        x_feature = body.get('xFeature')
        y_feature = body.get('yFeature')
        z_feature = body.get('zFeature')

        # Fetch all tunes
        tunes = Tune.objects.all()
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Use the processing pipeline to extract features dynamically
        tunes_extracted_features = processing_pipeline(abc_notations)

        # Initialize lists to hold extracted features for clustering
        x_data = []
        y_data = []
        z_data = []
        composers = []

        # Get selected features for each tune
        for tune_name, features in tunes_extracted_features.items():
            x_data.append(features.get(x_feature))
            y_data.append(features.get(y_feature))
            z_data.append(features.get(z_feature))
            composers.append(features.get('composer'))

        # Prepare data for clustering (3D points based on selected features)
        features_data = np.array(list(zip(x_data, y_data, z_data)))

        # Apply k-means clustering with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=0)
        clusters = kmeans.fit_predict(features_data)

        # Prepare the response data
        response_data = {
            'x': x_data,              # X-axis data
            'y': y_data,              # Y-axis data
            'z': z_data,              # Z-axis data
            'clusters': clusters.tolist(),  # Cluster assignment for each tune
            'composers': composers          # Composer names for hover info
        }

        return JsonResponse(response_data)
    
def calculate_feature_correlation(request):
    if request.method == 'POST':
        # Fetch all tunes
        tunes = Tune.objects.all()
        abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]

        # Use the processing pipeline to extract features for each tune
        tunes_extracted_features = processing_pipeline(abc_notations)

        # Create a DataFrame from the extracted features
        features_df = pd.DataFrame.from_dict(tunes_extracted_features, orient='index')

        # Exclude non-numeric columns (e.g., composer) from the correlation calculation
        numeric_features_df = features_df.select_dtypes(include=[float, int])

        # Calculate the correlation matrix and replace NaN values with 0
        correlation_matrix = numeric_features_df.corr().fillna(0)

        # Convert the correlation matrix to a format suitable for JSON response
        correlation_data = correlation_matrix.to_dict()

        return JsonResponse({
            'correlation_data': correlation_data,
            'feature_names': list(correlation_matrix.columns)  # Include feature names for labeling
        })
    
def search_tunes(request):
    if request.method == 'GET' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        search_type = request.GET.get('type')
        query = request.GET.get('query', '')

        #print(f"Search type: {search_type}, Query: {query}")

        # Filter tunes based on search type
        if search_type == 'name':
            tunes = Tune.objects.filter(name__icontains=query)
        elif search_type == 'composer':
            tunes = Tune.objects.filter(composer__icontains=query)
        else:
            tunes = Tune.objects.all()

        # Prepare the data to send back to the frontend
        tunes_data = [
            {
                'name': tune.name,
                'composer': tune.composer,
                'edit_url': reverse('tunes_edit', args=[tune.pk]),
                'delete_url': reverse('tunes_delete', args=[tune.pk])
            }
            for tune in tunes
        ]

        #print(tunes_data)

        return JsonResponse({'tunes': tunes_data})
    
def test_tunes(request):
    return render(request, 'test_tunes.html')

def get_tune_feature_values(request):
    if request.method == 'POST':
        # Get ABC notation from the request
        abc_notation = request.POST.get('abc_notation')

        if not abc_notation:
            return JsonResponse({'error': 'No ABC notation provided'}, status=400)

        # Process the ABC notation using processing_pipeline
        features = processing_pipeline([('unknown', 'unknown', abc_notation)])
        del features['unknown']['composer']
        
        # Return the feature values as JSON
        return JsonResponse(features['unknown'])
    
def get_tune_comparisons(request):
    abc_notation = request.GET.get('abc_notation', None)
    # List all features you want to calculate
    features = ["notes", "rests", "chords", "avg_pitch", "duration_sd"]
    feature_triplets = list(combinations(features, 3))

    # Process tunes
    tunes = Tune.objects.all()
    abc_notations = [(tune.name, tune.composer, tune.abc_notation) for tune in tunes]
    tunes_features = processing_pipeline(abc_notations)

    # Process uploaded tune features if abc_notation is provided
    uploaded_tune_features = None
    if abc_notation:
        uploaded_tune_features = processing_pipeline([('UserSubmitted', 'User', abc_notation)])['UserSubmitted']

    # Prepare data for plots
    data_for_triplets = {}
    for triplet in feature_triplets:
        triplet_data = {
            'x': [features.get(triplet[0], 0) for features in tunes_features.values()],
            'y': [features.get(triplet[1], 0) for features in tunes_features.values()],
            'z': [features.get(triplet[2], 0) for features in tunes_features.values()],
            'labels': [name for name in tunes_features.keys()],
            'composer': [features.get("composer", "unknown") for features in tunes_features.values()]
        }

        # Add uploaded tune's features to plot data if available
        if uploaded_tune_features:
            triplet_data['x'].append(uploaded_tune_features[triplet[0]])
            triplet_data['y'].append(uploaded_tune_features[triplet[1]])
            triplet_data['z'].append(uploaded_tune_features[triplet[2]])
            triplet_data['labels'].append("UserSubmitted")
            triplet_data['composer'].append("User")

        data_for_triplets["_".join(triplet)] = triplet_data

    # Return both features and plots in response
    return JsonResponse({
        'features': uploaded_tune_features if uploaded_tune_features else {},
        'plots': data_for_triplets
    })