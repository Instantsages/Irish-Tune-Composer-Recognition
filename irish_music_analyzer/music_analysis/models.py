from django.db import models

class Tune(models.Model):
    name = models.CharField(max_length=100)  # Tune's name
    composer = models.CharField(max_length=100)  # Composer's name
    abc_notation = models.TextField()  # ABC notation for the tune

    def __str__(self):
        return self.name  # String representation of the Tune object


class TuneAnalysis(models.Model):
    tune = models.ForeignKey(Tune, on_delete=models.CASCADE)  # Tune object
    avg_pitch = models.FloatField()  # Average pitch
    pitch_range = models.FloatField()  # Pitch range
    pitch_sd = models.FloatField()  # Pitch standard deviation
    avg_duration = models.FloatField()  # Average duration
    duration_range = models.FloatField()  # Duration range
    duration_sd = models.FloatField()  # Duration standard deviation
    avg_interval = models.FloatField()  # Average interval
    interval_range = models.FloatField()  # Interval range
    interval_sd = models.FloatField()  # Interval standard deviation
    contour_up = models.IntegerField()  # Number of upward contours
    contour_down = models.IntegerField()  # Number of downward contours
    note_density = models.FloatField()  # Note density
    syncopation_ratio = models.FloatField()  # Syncopation ratio
    different_rhythms = models.IntegerField()  # Number of different rhythms
    different_rhythms_ratio = models.FloatField()  # Different rhythms ratio

    def __str__(self):
        return f"{self.tune.name} Analysis"  # String representation of the TuneAnalysis object