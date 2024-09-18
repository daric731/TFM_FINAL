from django.shortcuts import render,redirect
from django.http import HttpResponse
from django import forms
from .models import Calendar
# Create your views here.
from .models import  Motor
import numpy as np
import pandas as pd
from django.http import HttpResponseBadRequest
from django.shortcuts import render
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from .forms import CalendarForm
from .models import Calendar, Motor
from django.contrib import messages
import plotly.express as px
import plotly.io as pio
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from django.shortcuts import render

from .models import Calendar

from datetime import datetime
from calendar import monthrange, month_name
from django.shortcuts import render, redirect
from django.utils.dateformat import format
from calendar import monthrange
from calendar import monthrange, Calendar as CalendarModule
from .models import Calendar


from .models import Calendar as CalendarModel

from django.shortcuts import render, redirect
from django.http import HttpResponseBadRequest


from django.utils.dateparse import parse_datetime

from .models import Calendar as CalendarModel, Motor

from django.utils.dateformat import format
from django.utils import timezone
from calendar import Calendar as PyCalendar, monthrange
from datetime import datetime
from calendar import Calendar as PyCalendar, monthrange
from django.utils.dateparse import parse_datetime
from django.utils import timezone
from django.http import HttpResponseBadRequest
from django.shortcuts import render

from .models import Calendar 
def analysis_view(request):
    # Initialize context variables
    context = {
        'ids': Motor.objects.values_list('id', flat=True).distinct(),
        'entries': request.session.get('entries', []),
        'plots': request.session.get('plots', []),
        'error': request.session.get('error', None),
        'num_entries': request.session.get('num_entries', 10),
        'selected_id': request.session.get('selected_id', None),
        'selected_id1': request.session.get('selected_id1', None),
        'selected_id2': request.session.get('selected_id2', None),
        'selected_sensor1': request.session.get('selected_sensor1', None),
        'selected_sensor2': request.session.get('selected_sensor2', None),
        'columns': [f's{i}' for i in range(1, 22)],
    }
    return render(request, 'analysis/analysis.html', context)

def data_analysis(request):
    # Initialize context
    context = {
        'entries': request.session.get('entries', []),
        'ids': Motor.objects.values_list('id', flat=True).distinct(),
        'plots': request.session.get('plots', []),
        'columns': [f's{i}' for i in range(1, 22)],
        'selected_id1': request.POST.get('plot_id1', request.session.get('selected_id1')),
        'selected_id2': request.POST.get('plot_id2', request.session.get('selected_id2')),
        'selected_sensor1': request.POST.get('plot_sensor1', request.session.get('selected_sensor1')),
        'selected_sensor2': request.POST.get('plot_sensor2', request.session.get('selected_sensor2')),
        'selected_id': request.POST.get('selected_id', request.session.get('selected_id')),
        'num_entries': int(request.POST.get('num_entries', request.session.get('num_entries', 10))),
        'error': None,
    }

    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'fetch_entries':
            if context['selected_id']:
                context['entries'] = list(Motor.objects.filter(id=context['selected_id']).order_by('-id')[:context['num_entries']].values())
            else:
                context['entries'] = list(Motor.objects.all().order_by('-id')[:context['num_entries']].values())
        
        elif action == 'generate_plots':
            context['plots'] = []  # Clear previous plots

            if context['selected_id1'] and context['selected_id2'] and context['selected_sensor1']:
                data = Motor.objects.filter(id__in=[context['selected_id1'], context['selected_id2']]).order_by('cycle')
                df = pd.DataFrame(list(data.values()))

                # Generate plot for the first sensor
                if context['selected_sensor1'] in df.columns:
                    fig1 = px.line(df, x='cycle', y=context['selected_sensor1'], color='id',
                                   title=f'{context["selected_sensor1"]} vs Cycle for IDs {context["selected_id1"]} and {context["selected_id2"]}',
                                   labels={'cycle': 'Cycle', context['selected_sensor1']: context['selected_sensor1']})
                    fig1.update_layout(legend_title='ID')
                    context['plots'].append(pio.to_html(fig1, full_html=False))

                # Generate plot for the second sensor
                if context['selected_sensor2'] in df.columns:
                    fig2 = px.line(df, x='cycle', y=context['selected_sensor2'], color='id',
                                   title=f'{context["selected_sensor2"]} vs Cycle for IDs {context["selected_id1"]} and {context["selected_id2"]}',
                                   labels={'cycle': 'Cycle', context['selected_sensor2']: context['selected_sensor2']})
                    fig2.update_layout(legend_title='ID')
                    context['plots'].append(pio.to_html(fig2, full_html=False))

            if not context['plots']:
                context['error'] = 'Please ensure both IDs and sensors are selected and valid for plotting.'

        # Save the current selections and plots to the session
        request.session['entries'] = context['entries']
        request.session['plots'] = context['plots']
        request.session['selected_id1'] = context['selected_id1']
        request.session['selected_id2'] = context['selected_id2']
        request.session['selected_sensor1'] = context['selected_sensor1']
        request.session['selected_sensor2'] = context['selected_sensor2']
        request.session['selected_id'] = context['selected_id']
        request.session['num_entries'] = context['num_entries']
        request.session['error'] = context['error']

    else:
        context['entries'] = list(Motor.objects.all().order_by('-id')[:10].values())

    return render(request, 'analysis/analysis.html', context)
def test_view(request):
    return HttpResponse("This is a test view.")

def test_template_view(request):
    return render(request, 'analysis/analysis.html')

def home(request):
    # Your view logic here
    return render(request, 'home.html')



def calendar_view(request):
    # Initialize form
    form = CalendarForm(request.POST or None)
    events = []

    if request.method == 'POST':
        if 'save_entry' in request.POST:
            if form.is_valid():
                try:
                    form.save()
                    messages.success(request, 'Calendar entry saved successfully!')
                    return redirect('calendar_view')  # Redirect to avoid resubmission on refresh
                except Exception as e:
                    pass        
        elif 'display_events' in request.POST:
            # Get the events without requiring form submission
            events = Calendar.objects.all()

    # Render the form and events
    return render(request, 'calendar/calendar.html', {'form': form, 'events': events})



# Paths to the model files
model1_path = 'C:/Users/daric/OneDrive/Desktop/tfm_v2/models/model_rf_binary_smote.joblib'
model3_path = 'C:/Users/daric/OneDrive/Desktop/tfm_v2/models/model3_lstm_RUL.keras'
rf_model_path = 'C:/Users/daric/OneDrive/Desktop/tfm_v2/models/model_rf_multiclassifier_smote.joblib'

# Load models
try:
    model3_lstm_RUL = tf.keras.models.load_model(model3_path)
    print(f"model3_lstm_RUL loaded successfully from {model3_path}.")
except Exception as e:
    print(f"Error loading model3_lstm_RUL: {e}")

try:
    model_rf_binary_smote = joblib.load(model1_path)
    print(f"model_rf_binary_smote loaded successfully from {model1_path}.")
except Exception as e:
    print(f"Error loading model_rf_binary_smote: {e}")

try:
    model_rf_multiclassifier_smote = joblib.load(rf_model_path)
    print(f"model_rf_multiclassifier_smote loaded successfully from {rf_model_path}.")
except Exception as e:
    print(f"Error loading model_rf_multiclassifier_smote: {e}")

sequence_length = 50
sequence_cols = [
    'setting1', 'setting2', 'setting3', 'cycle_norm',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
    's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
    's18', 's19', 's20', 's21'
]

def prepare_sequences_for_prediction(df, sequence_length, sequence_cols):
    sequences = []
    # Create sequences for LSTM prediction
    for i in range(len(df) - sequence_length + 1):
        seq = df[sequence_cols].iloc[i:i + sequence_length].values
        sequences.append(seq)
    return np.array(sequences)

def make_lstm_predictions(model, df, sequence_length, sequence_cols):
    if len(df) < sequence_length:
        raise ValueError(f"Not enough data to create sequences. Need at least {sequence_length} rows.")
    X_sequences = prepare_sequences_for_prediction(df, sequence_length, sequence_cols)
    lstm_predictions = model.predict(X_sequences)
    if lstm_predictions.ndim == 3:
        lstm_predictions = np.squeeze(lstm_predictions, axis=1)
    df['RUL_PREDICTION'] = np.nan
    for i in range(len(lstm_predictions)):
        df.loc[i + sequence_length - 1, 'RUL_PREDICTION'] = lstm_predictions[i]
    return df

def make_rf_predictions(df, rf_binary_model, rf_multiclass_model, sequence_cols):
    rf_features = df[sequence_cols]
    binary_predictions = rf_binary_model.predict(rf_features)
    multiclass_predictions = rf_multiclass_model.predict(rf_features)
    df['BINARY_CLASSIFICATION'] = binary_predictions
    df['MULTICLASS_CLASSIFICATION'] = multiclass_predictions
    return df
def predictions_view(request):
    context = {
        'ids': Motor.objects.values_list('id', flat=True).distinct(),
        'entries': [],
        'selected_id': None,
        'transformed_df': None,
        'has_data': False  # Flag to check if DataFrame has data
    }

    if request.method == 'POST':
        selected_id = request.POST.get('selected_id')
        if selected_id:
            df = pd.DataFrame(list(Motor.objects.filter(id=selected_id).values()))
            df['cycle_norm'] = df['cycle']
            cols_to_keep = sequence_cols + ['id', 'cycle']
            transformed_df = df[cols_to_keep].copy()
            
            try:
                # Apply LSTM predictions
                transformed_df = make_lstm_predictions(model3_lstm_RUL, transformed_df, sequence_length, sequence_cols)
                
                # Apply RF predictions
                transformed_df = make_rf_predictions(transformed_df, model_rf_binary_smote, model_rf_multiclassifier_smote, sequence_cols)

                # Rename prediction columns
                transformed_df.rename(columns={
                    'RUL_PREDICTION': 'RUL',
                    'BINARY_CLASSIFICATION': 'Binary Classification',
                    'MULTICLASS_CLASSIFICATION': 'Multiclass Classification'
                }, inplace=True)
                
                # Drop the 'id' column
                transformed_df.drop(columns=['id'], inplace=True)

                context['transformed_df'] = transformed_df
                context['selected_id'] = selected_id
                context['has_data'] = not transformed_df.empty  # Update flag based on DataFrame content

            except ValueError as e:
                print(f"Error: {e}")

    return render(request, 'predictions/predictions.html', context)