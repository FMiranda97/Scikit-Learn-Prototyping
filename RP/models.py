from django.db import models


# Create your models here.
class FilterSet(models.Model):
    app_label = 'RP2021'
    info = models.TextField()
    report = models.TextField(null=True)
    name = models.CharField(max_length=64)


class Classifier(models.Model):
    app_label = 'RP2021'
    model = models.TextField()
    name = models.CharField(max_length=64)
