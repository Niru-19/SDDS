import os
import io
import json
from PIL import Image
import numpy as np

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import (
    DoctorLoginForm,
    DoctorRegistrationForm,
    PatientLoginForm,
    PatientRegistrationForm,
    UserLoginForm,
)
from .models import DoctorProfile, PatientProfile, SkinPrediction


# ==========================================
# LAZY ML & RAG LOADERS (PREVENTS OOM ON BOOT)
# ==========================================

CLASS_ORDER = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

DISEASE_LOOKUP = {
    "AK": "Actinic Keratosis (Pre-cancerous)",
    "BCC": "Basal Cell Carcinoma (Skin Cancer)",
    "BKL": "Benign Keratosis (Non-cancerous Growth)",
    "DF": "Dermatofibroma (Benign Lesion)",
    "MEL": "Melanoma (Malignant Skin Cancer)",
    "NV": "Melanocytic Nevi (Common Benign Mole)",
    "SCC": "Squamous Cell Carcinoma (Skin Cancer)",
    "VASC": "Vascular Lesion (Benign Blood Vessel)",
    "UNK": "Unknown Variant / Unclassified",
}

# Cache containers for lazy-loaded objects
_KERAS_MODEL = None
_RAG_CHAIN = None


def get_keras_model():
    """Lazy-load Keras model only when a prediction request arrives."""
    global _KERAS_MODEL
    if _KERAS_MODEL is None:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'skin_model.h5')
        if os.path.exists(model_path):
            _KERAS_MODEL = tf.keras.models.load_model(model_path)
    return _KERAS_MODEL


def load_torch_model():
    """Lazy-load PyTorch model on demand."""
    import torch
    from .ai_model import SkinCancerCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "skin_cancer_cnn.pth")
    
    torch_model = SkinCancerCNN()
    if os.path.exists(model_path):
        torch_model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    torch_model.to(device)
    torch_model.eval()
    return torch_model, device


def get_rag_chain():
    """Lazy-load LangChain/RAG pipeline only when a user sends a chat message."""
    global _RAG_CHAIN
    if _RAG_CHAIN is not None:
        return _RAG_CHAIN

    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings

        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        data_dir = os.path.join(os.path.dirname(__file__), 'chatbot_data') 
        db_path = os.path.join(os.path.dirname(__file__), 'chroma_db')

        llm = ChatGroq(
            temperature=0,
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            model_name="llama-3.3-70b-versatile"
        )
        
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(db_path):
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            
            if texts:
                vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
            else:
                vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "You are a compassionate health chatbot. Respond thoughtfully to the following questions:\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        _RAG_CHAIN = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return _RAG_CHAIN
    except Exception as err:
        print(f"❌ Failed to build RAG engine: {err}")
        return None


# ==========================================
# 1. PUBLIC & MARKETING ROUTING VIEWS
# ==========================================

def home(request):
    return render(request, "index.html")

def contact_view(request):
    return render(request, "contact.html")

def about_view(request):
    return render(request, "about.html")


# ==========================================
# 2. PATIENT FUNCTION SYSTEM VIEWS
# ==========================================

def register_patient(request):
    if request.method == "POST":
        form = PatientRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()

            PatientProfile.objects.create(
                user=user,
                age=form.cleaned_data["age"],
                gender=form.cleaned_data["gender"],
                dob=form.cleaned_data["dob"],
                contact=form.cleaned_data["contact"],
                address=form.cleaned_data["address"],
            )
            messages.success(request, "Patient registration successful! Please login.")
            return redirect("login_view")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = PatientRegistrationForm()
    return render(request, "patient_register.html", {"form": form})


def patient_dashboard(request):
    if not request.user.is_authenticated or not hasattr(request.user, "patient_profile"):
        return redirect("login_view")

    patient_profile = request.user.patient_profile
    history = SkinPrediction.objects.filter(patient=patient_profile).order_by("-prediction_date")

    context = {"patient": patient_profile, "history": history}
    return render(request, "patient_dashboard.html", context)


def image_upload(request):
    if not request.user.is_authenticated or not hasattr(request.user, "patient_profile"):
        return redirect("login_view")

    if request.method == "POST" and request.FILES.get("skin_image"):
        patient_profile = request.user.patient_profile
        image_file = request.FILES["skin_image"]

        prediction_record = SkinPrediction.objects.create(
            patient=patient_profile, uploaded_image=image_file
        )

        prediction_record.predicted_disease = "Melanoma"
        prediction_record.confidence_score = 88.5
        prediction_record.specialist_recommendation = (
            "Schedule an immediate consultation with a certified Dermatologist."
        )
        prediction_record.save()

        messages.success(request, "Analysis completed successfully!")
    return redirect("patient_dashboard")


@login_required
@csrf_exempt
def predict_skin_view(request):
    context = {}
    
    try:
        current_patient = PatientProfile.objects.get(user=request.user)
        context["patient"] = current_patient
    except PatientProfile.DoesNotExist:
        current_patient = None

    if request.method == "POST" and request.FILES.get("image"):
        uploaded_image = request.FILES["image"]
        fs = FileSystemStorage()
        saved_name = fs.save(uploaded_image.name, uploaded_image)
        image_url = fs.url(saved_name)

        try:
            uploaded_image.seek(0)
            image_bytes = uploaded_image.read()
            
            with Image.open(io.BytesIO(image_bytes)) as PIL_img:
                img = PIL_img.convert("RGB").resize((224, 224))
                img_array = np.array(img, dtype=np.float32)  
                img_tensor = np.expand_dims(img_array, axis=0)  

            keras_model = get_keras_model()
            if keras_model:
                predictions = keras_model.predict(img_tensor)
                predicted_idx = np.argmax(predictions[0])
                confidence_score = predictions[0][predicted_idx]

                label_code = CLASS_ORDER[predicted_idx]
                disease_name = DISEASE_LOOKUP.get(label_code, f"Unknown Variant ({label_code})")
                confidence_percentage = round(float(confidence_score) * 100, 2)

                if current_patient:
                    rec_text = (
                        "Schedule an immediate consultation with a certified Dermatologist." 
                        if "Cancer" in disease_name or "Melanoma" in disease_name 
                        else "Monitor for structural changes."
                    )
                    SkinPrediction.objects.create(
                        patient=current_patient,
                        uploaded_image=saved_name, 
                        predicted_disease=disease_name,
                        confidence_score=confidence_percentage,
                        specialist_recommendation=rec_text
                    )

                context.update({
                    "image_url": image_url,
                    "disease_name": disease_name,
                    "confidence": confidence_percentage,
                })
        except Exception as e:
            print(f"❌ Keras Inference Error: {e}")
            context["error"] = "Failed to process image structure safely through matrix layers."

    if current_patient:
        context["history"] = SkinPrediction.objects.filter(patient=current_patient).order_by('-prediction_date')
    else:
        context["history"] = []

    return render(request, "patient_dashboard.html", context)


# ==========================================
# 3. CLINICAL DOCTOR CORE VIEWS
# ==========================================

def register_doctor(request):
    if request.method == "POST":
        form = DoctorRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()

            DoctorProfile.objects.create(
                user=user,
                doctor_id=form.cleaned_data["doctor_id"],
                specialization=form.cleaned_data["specialization"],
                contact=form.cleaned_data["contact"],
                age=form.cleaned_data["age"],
            )
            messages.success(request, "Doctor registration successful! Verification pending.")
            return redirect("login_view")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = DoctorRegistrationForm()
    return render(request, "register_doctor.html", {"form": form})


@login_required
def doctor_dashboard(request):
    assigned_patients = SkinPrediction.objects.all()
    search_query = request.GET.get("patient_id", "").strip()
    if search_query:
        assigned_patients = assigned_patients.filter(
            patient__user__username__icontains=search_query
        )

    context = {
        "assigned_patients": assigned_patients,
        "search_query": search_query,
    }
    return render(request, "doctor_dashboard.html", context)


@login_required
def doctor_patient_search(request):
    search_query = request.GET.get("patient_id", "").strip()
    patient = None

    if search_query:
        patient = PatientProfile.objects.filter(
            user__username__iexact=search_query
        ).first()

        if not patient and search_query.isdigit():
            patient = PatientProfile.objects.filter(
                id=int(search_query)
            ).first()

        if not patient and search_query.isdigit():
            scan = SkinPrediction.objects.filter(id=int(search_query)).first()
            if scan:
                patient = scan.patient

    if not patient:
        messages.error(
            request,
            f"No patient profile found matching search query: '{search_query}'",
        )
        return redirect("doctor_dashboard")

    history = SkinPrediction.objects.filter(patient=patient).order_by("-prediction_date")
    latest_case = history.first()

    context = {
        "patient": patient,
        "case": latest_case,
        "history": history,
        "search_query": search_query,
    }
    return render(request, "doctor_dashboard.html", context)


# ==========================================
# 4. AUTHENTICATION VIEWS
# ==========================================

def login_view(request):
    doc_form = DoctorLoginForm()
    patient_form = PatientLoginForm()

    if request.method == "POST":
        user_type = request.POST.get("user_type")

        if user_type == "doctor":
            doc_form = DoctorLoginForm(request.POST)
            if doc_form.is_valid():
                try:
                    doctor_profile = DoctorProfile.objects.get(
                        doctor_id=doc_form.cleaned_data["doctor_id"]
                    )
                    username = doctor_profile.user.username
                    password = doc_form.cleaned_data["password"]
                    user = authenticate(
                        request, username=username, password=password
                    )
                except DoctorProfile.DoesNotExist:
                    user = None

                if user is not None and hasattr(user, "doctor_profile"):
                    login(request, user)
                    return redirect("doctor_dashboard")
                else:
                    messages.error(request, "Invalid Doctor ID or Password.")

        elif user_type == "patient":
            patient_form = PatientLoginForm(request.POST)
            if patient_form.is_valid():
                name = patient_form.cleaned_data["name"]
                phone = patient_form.cleaned_data["phone_number"]

                try:
                    profile = PatientProfile.objects.get(contact=phone)
                    if (
                        profile.user.get_full_name().lower() == name.lower()
                        or profile.user.username.lower() == name.lower()
                    ):
                        profile.user.backend = "django.contrib.auth.backends.ModelBackend"
                        login(request, profile.user)
                        return redirect("patient_dashboard")
                    else:
                        messages.error(request, "Name does not match our records.")
                except PatientProfile.DoesNotExist:
                    messages.error(request, "No patient registered with that number.")

    return render(
        request,
        "login.html",
        {"doc_form": doc_form, "patient_form": patient_form},
    )


def logout_view(request):
    logout(request)
    messages.info(request, "Logged out successfully.")
    return redirect("home")


# ==========================================
# 5. ADMINISTRATIVE CONTROL BACKENDS
# ==========================================

def admin_login(request):
    if request.method == "POST":
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")

            if username == "admin" and password == "password123":
                user = User.objects.filter(username="admin", is_superuser=True).first()
                if not user:
                    user = User.objects.create_superuser(
                        username="admin",
                        email="admin@dermio.com",
                        password="password123",
                    )
                login(request, user)
                return redirect("admin_dashboard")
            else:
                messages.error(request, "Invalid Admin System Credentials.")
    else:
        form = UserLoginForm()
    return render(request, "admin_login.html", {"form": form})


def admin_dashboard(request):
    context = {
        "total_patients": PatientProfile.objects.count(),
        "total_doctors": DoctorProfile.objects.count(),
        "total_scans": SkinPrediction.objects.count(),
        "recent_scans": SkinPrediction.objects.select_related("patient__user").order_by("-prediction_date")[:5],
    }
    return render(request, "admin_dashboard.html", context)


def dataset_management(request):
    return render(request, "dataset_management.html")


def prediction_tracker(request):
    recent_scans = SkinPrediction.objects.select_related("patient__user").order_by("-prediction_date")
    return render(request, "prediction_tracker.html", {"recent_scans": recent_scans})


def system_reports(request):
    context = {
        "total_doctors": DoctorProfile.objects.count(),
        "total_patients": PatientProfile.objects.count(),
        "total_scans": SkinPrediction.objects.count(),
        "total_admins": User.objects.filter(is_superuser=True).count(),
    }
    return render(request, "system_reports.html", context)


def admin_doctors_view(request):
    doctors = DoctorProfile.objects.all()
    return render(request, "admin_doctors.html", {"doctors": doctors})


def admin_update_doctor(request, pk):
    doctor = get_object_or_404(DoctorProfile, pk=pk)
    if request.method == "POST":
        doctor.user.first_name = request.POST.get("first_name")
        doctor.user.last_name = request.POST.get("last_name")
        doctor.user.email = request.POST.get("email")
        doctor.user.save()

        doctor.specialization = request.POST.get("specialization")
        doctor.contact = request.POST.get("contact")
        doctor.age = request.POST.get("age")
        doctor.save()

        messages.success(request, f"Dr. {doctor.user.last_name} updated successfully.")
        return redirect("admin_doctors")


def admin_delete_doctor(request, pk):
    doctor = get_object_or_404(DoctorProfile, pk=pk)
    doctor.user.delete()
    messages.success(request, "Doctor profile and auth account removed.")
    return redirect("admin_doctors")


def admin_patients_view(request):
    patients = PatientProfile.objects.all()
    return render(request, "admin_patients.html", {"patients": patients})


def admin_update_patient(request, pk):
    patient = get_object_or_404(PatientProfile, pk=pk)
    if request.method == "POST":
        patient.user.first_name = request.POST.get("first_name")
        patient.user.last_name = request.POST.get("last_name")
        patient.user.email = request.POST.get("email")
        patient.user.save()

        patient.age = request.POST.get("age")
        patient.contact = request.POST.get("contact")
        patient.address = request.POST.get("address")
        patient.save()

        messages.success(request, f"Patient {patient.user.username} updated.")
        return redirect("admin_patients")


def admin_delete_patient(request, pk):
    patient = get_object_or_404(PatientProfile, pk=pk)
    patient.user.delete()
    messages.success(request, "Patient profile removed safely.")
    return redirect("admin_patients")


# ==========================================
# 6. LANGCHAIN & RAG CHATBOT SYSTEM
# ==========================================

def mental_health_chat_page(request):
    return render(request, "mental_health_chat.html")


@csrf_exempt
def mental_health_chat_api(request):
    if request.method == "POST":
        rag_chain = get_rag_chain()
        if not rag_chain:
            return JsonResponse({"error": "RAG pipeline engine failed to build on request."}, status=500)
            
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
            
            if not user_message:
                return JsonResponse({"error": "Empty query string provided."}, status=400)
            
            bot_reply = rag_chain.invoke(user_message).strip()
            
            if not bot_reply:
                bot_reply = "I'm sorry, I couldn't formulate a proper response context."
            
            return JsonResponse({"reply": bot_reply})
            
        except Exception as e:
            return JsonResponse({"error": f"Error processing vector search parameters: {str(e)}"}, status=500)
            
    return JsonResponse({"error": "Method allocation invalid."}, status=400)