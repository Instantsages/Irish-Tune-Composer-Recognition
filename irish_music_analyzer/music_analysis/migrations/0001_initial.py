# Generated by Django 5.1.1 on 2024-10-06 22:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Tune',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('composer', models.CharField(max_length=100)),
                ('abc_notation', models.TextField()),
            ],
        ),
    ]