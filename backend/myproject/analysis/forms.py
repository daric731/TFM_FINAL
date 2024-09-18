from django import forms
from .models import Calendar, Motor





class CalendarForm(forms.ModelForm):
    class Meta:
        model = Calendar
        fields = ['date_time', 'motor_id', 'entry_text']
        widgets = {
            'date_time': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'motor_id': forms.NumberInput(attrs={'min': 1, 'max': 100}),
            'entry_text': forms.Textarea(attrs={'rows': 4}),
        }