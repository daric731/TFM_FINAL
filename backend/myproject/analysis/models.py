from django.db import models

# Create your models here.

class Motor(models.Model):
    cycle = models.IntegerField()
    setting1 = models.FloatField()
    setting2 = models.FloatField()
    setting3 = models.FloatField()
    s1 = models.FloatField()
    s2 = models.FloatField()
    s3 = models.FloatField()
    s4 = models.FloatField()
    s5 = models.FloatField()
    s6 = models.FloatField()
    s7 = models.FloatField()
    s8 = models.FloatField()
    s9 = models.FloatField()
    s10 = models.FloatField()
    s11 = models.FloatField()
    s12 = models.FloatField()
    s13 = models.FloatField()
    s14 = models.FloatField()
    s15 = models.FloatField()
    s16 = models.FloatField()
    s17 = models.IntegerField()
    s18 = models.IntegerField()
    s19 = models.FloatField()
    s20 = models.FloatField()
    s21 = models.FloatField()


    class Meta:
        db_table = 'motor'



class Calendar(models.Model):
    id = models.AutoField(primary_key=True)
    date_time = models.DateTimeField()
    motor_id = models.IntegerField()  # Use IntegerField for Motor ID
    entry_text = models.TextField(max_length=500)

    def __str__(self):
        return f"{self.date_time} - {self.entry_text[:20]}"