from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
    path('register/patient/', views.register_patient, name='register_patient'),
    path('register/doctor/', views.register_doctor, name='register_doctor'),
    path('login/', views.login_view, name='login_view'),
    path('logout/', views.logout_view, name='logout_view'),
    path('dashboard/patient/', views.patient_dashboard, name='patient_dashboard'),
    path('upload-image/', views.image_upload, name='image_upload'),
    path('doctor/dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('doctor/search-case/', views.doctor_patient_search, name='doctor_patient_search'),
    path('admin-login/', views.admin_login, name='admin_login'), 
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('dataset/', views.dataset_management, name='dataset_management'),
    path('predictions/', views.prediction_tracker, name='predictions_tracker'),
    path('reports/', views.system_reports, name='system_reports'),
    path('predict/', views.predict_skin_view, name='predict_skin'),
    path('admin-hub/doctors/', views.admin_doctors_view, name='admin_doctors'),
    path('admin-hub/doctors/update/<int:pk>/', views.admin_update_doctor, name='admin_update_doctor'),
    path('admin-hub/doctors/delete/<int:pk>/', views.admin_delete_doctor, name='admin_delete_doctor'),
    path('admin-hub/patients/', views.admin_patients_view, name='admin_patients'),
    path('admin-hub/patients/update/<int:pk>/', views.admin_update_patient, name='admin_update_patient'),
    path('admin-hub/patients/delete/<int:pk>/', views.admin_delete_patient, name='admin_delete_patient'),
    
    # Mental Health RAG Chatbot Routes
    path('mindcare/', views.mental_health_chat_page, name='mental_health_chat_page'),
    path('api/mental-health-chat/', views.mental_health_chat_api, name='mental_health_chat_api'),
]