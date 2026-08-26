from django.db import models
from django.contrib.auth.models import User

# 1. DOCTOR PROFILE
class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile')
    doctor_id = models.CharField(max_length=20, unique=True)
    specialization = models.CharField(max_length=100)
    contact = models.CharField(max_length=15)
    age = models.IntegerField()

    def __str__(self):
        return f"Dr. {self.user.get_full_name() or self.user.username}"


# 2. PATIENT PROFILE
class PatientProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='patient_profile')
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    dob = models.DateField(verbose_name="Date of Birth")
    contact = models.CharField(max_length=15)
    address = models.TextField()

    # 💡 FIXED: Stripped away the trailing "rPr" typo here
    def __str__(self):
        return self.user.get_full_name() or self.user.username


# 3. SKIN DISEASE PREDICTION (From Patient Upload)
class SkinPrediction(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name='predictions')
    uploaded_image = models.ImageField(upload_to='skin_leasions/')
    predicted_disease = models.CharField(max_length=100, blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    specialist_recommendation = models.CharField(max_length=150, blank=True, null=True)
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.patient} - {self.prediction_date.strftime('%Y-%m-%d')}"


# 4. DOCTOR DIAGNOSIS (From Doctor Dashboard)
class DoctorDiagnosis(models.Model):
    prediction = models.OneToOneField(SkinPrediction, on_delete=models.CASCADE, related_name='diagnosis')
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.SET_NULL, null=True, related_name='given_diagnoses')
    diagnosis_details = models.TextField()
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Diagnosis by {self.doctor} for case #{self.prediction.id}"
    

from django.db import models
from django.contrib.auth.models import User
from .models import PatientProfile  # Ensure this is imported

class ChatMessage(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name='chat_messages')
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('bot', 'Bot')])
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']  # Keeps conversation in chronological order

    def __str__(self):
        return f"{self.sender.capitalize()}: {self.message[:30]}"