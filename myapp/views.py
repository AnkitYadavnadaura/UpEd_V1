from django.shortcuts import render, redirect, get_object_or_404
from .models import CustomUser, Gen_Content, CoursePlan, Module, Subtopic, CoursePlanOnSyllabus, ModuleOnSyllabus, SubtopicOnSyllabus
from django.contrib.auth import get_user_model, authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse, HttpResponse
from django.core.mail import send_mail
import random
from django.utils import timezone
import datetime
from datetime import timedelta
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os
from dotenv import load_dotenv
import json
import requests
from django.urls import reverse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from . import filestorage
import tensorflow as tf

# Create your views here.
# pep
# hardikjainharsora@gmail.com
# pep1234
User = get_user_model()
import traceback
import logging

logger = logging.getLogger(__name__)

def log_exception(e):
    """Print and log the full traceback for any exception."""
    error_message = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print(error_message)  # This will show up in `runserver` console
    logger.error(error_message)  # Also log to Django's logger if configured

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
# google_api_key = "AIzaSyBZ1BFc6nKOKFllAYc7CxtFKnMIeHjA9GU"
chat = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0, google_api_key=google_api_key)


def index(request):
    return render(request, 'index.html')


otp_storage = {}

def register(request):
    if request.method == 'POST':
        # name = request.POST['name']
        # phone_number = request.POST['number']
        # email = request.POST['email']
        # age = request.POST['age']
        # gender = request.POST['gender']
        # password = request.POST['password']
        # conf_pass = request.POST['confpass']

        # if password != conf_pass:
        #     return render(request, 'signup.html', {'error': 'Passwords do not match.'})
        data = json.loads(request.body.decode('utf-8'))
        name = data.get('name')
        email = data.get('email')
        phone_number = data.get('number')
        age = data.get('age')
        gender = data.get('gender')
        password = data.get('password')
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({"success": False, "error": "Email already exists"}, status=400)
        if User.objects.filter(phone_number=phone_number).exists():
            return JsonResponse({"success": False, "error": "mobile already exists"}, status=400)
        
        otp = random.randint(100000, 999999)
        otp_storage[email] = {
            'otp': otp,
            'name': name,
            'phone_number': phone_number,
            'email': email,
            'age': age,
            'gender': gender,
            'password': password,
            'otp_time': timezone.now()
        }
        send_otp_mail(name, email, otp)
        return JsonResponse({"success": True, "status": "OTP sent to Mail"}, status=200)
    if request.user.is_authenticated:
        return JsonResponse({"success": True, "error": "already LoggedIn"}, status=200)
    return JsonResponse({"success": False, "error": "Invalid request"}, status=200)

def verify_otp(request, email):
    if request.user.is_authenticated:
        return JsonResponse({"success": True, "error": "already LoggedIn"}, status=200)
    otp_data = otp_storage.get(email)
    if otp_data is None:
        return JsonResponse({"success": False, "error": "Registor"}, status=400)
    otp_time = otp_data['otp_time']
    otp_expired = timezone.now() > otp_time + timedelta(minutes=5)
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        entered_otp = data.get('otp')
        # entered_otp = request.POST.get('entered_otp')
        if (not otp_expired and str(otp_data['otp']) == entered_otp):
            user_create = User.objects.create_user(
                username=email,
                email=email,
                name=otp_data['name'],
                password=otp_data['password'],
                gender=otp_data['gender'],
                age=otp_data['age'],
                phone_number=otp_data['phone_number']
            )
            login(request, user_create)
            return JsonResponse({"success": True, "error": "already LoggedIn"}, status=200)
        else:
            return JsonResponse({"success": False, "error": "Invalid OTP"}, status=200)
    return JsonResponse({"success": False, "error": "Fuck U"}, status=200)

def resend_otp(request, email):
    if request.user.is_authenticated:
        return JsonResponse({"success": True, "error": "already LoggedIn"}, status=200)
    otp_data = otp_storage.get(email)
    if otp_data is None:
        return JsonResponse({"success": False, "error": "Registor"}, status=400)
    otp = random.randint(100000, 999999)
    otp_data['otp_time'] = timezone.now()
    otp_data['otp'] = otp
    otp_storage[email] = otp_data
    send_otp_mail(otp_data['name'], email, otp)
    return JsonResponse({"success": True, "status": "OTP sent to Mail"}, status=200)

def send_otp_mail(name, email, otp):
    subject = 'Your OTP for email verification'
    message = f'Hey, {name}.\n Welcome to PEP. Your OTP for email verification is: {otp}.'
    from_email = 'heelo@gmail.com'
    recipient_list = [email]
    send_mail(subject, message, from_email, recipient_list, fail_silently=False)
from itertools import chain
@login_required(login_url='/login/')
def dashboard(request):
    # courses = list(chain(CoursePlan.objects.filter(user=request.user) , CoursePlanOnSyllabus.objects.filter(user=request.user)))
    # if not courses:
        # return render(request, "dashboard.html", {"error": "There is no such course present here."})
    safe_email = request.user.email.replace("@", "_at_")
    filename = f"{safe_email}.json"
    user_data = filestorage.load_data(filename)
    return JsonResponse({"success": True,"user":{"name":request.user.name, "Email":request.user.email}, "user_data": user_data}, status=200)

@login_required(login_url='/login/')
def getcourseslist(request,e):
    data= filestorage.load_data(e)
    return JsonResponse({"success": True,"courseslist":data}, status=200)
def check_email(request):
    email = request.GET.get('email', None)
    data = {
        'is_taken': User.objects.filter(email__iexact=email).exists()
    }
    return JsonResponse(data)

def loginuser(request):
    error = ''
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        identifier = data.get('identifier')
        password = data.get('pass12')
        user_login = authenticate(request, username=identifier, password=password)
        if user_login is not None:
            login(request, user_login)
        else:
            error = 'Invalid username or password'
    if request.user.is_authenticated:
        return JsonResponse({"success": True,"userlog":False}, status=200)
    return JsonResponse({"success": True,"userlog":True,"error":error}, status=200)

def logoutuser(request):
    logout(request)
    return redirect('login')

from slugify import slugify
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from django.views.decorators.csrf import csrf_protect
from langchain_community.document_loaders import PyMuPDFLoader

class ModulePydantic(BaseModel):
    topic: str
    subtopics: list[str]

class CoursePlanPydantic(BaseModel):
    course_name: str
    level: str
    language: str
    goal: str
    modules: list[ModulePydantic]

import re
from fuzzywuzzy import fuzz
def normalize_course_name(name):
    return re.sub(r'[^a-zA-Z0-9 ]', '', name.lower().strip())

from difflib import SequenceMatcher

def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def get_duplicate_course(course_name, db):
    print(course_name)
    norm_name = normalize_course_name(course_name)
    data = filestorage.load_data("courses.json")["courses"]
    user_courses = data
    for entry in user_courses:
        entry_norm = normalize_course_name(entry)
        score = fuzz.ratio(norm_name, entry_norm)
        if score > 80:
            return entry
    return None

@login_required(login_url='/login/')
def submit_preferences(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        error = ''
        course = data.get('course')
        goals = data.get('goals', 'To learn')
        level = data.get('level', 'advanced')
        language = data.get('language', 'English')
        print(course)
        dup = get_duplicate_course(course+''+goals + '' + level + '' + language, CoursePlan.objects.filter(user=request.user))
        if dup:
            return JsonResponse({"success": True,"data":"duplicate","error":error}, status=200)
        parser = PydanticOutputParser(pydantic_object=CoursePlanPydantic)
        format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        prompt_template = PromptTemplate.from_template("""
You are a course planning assistant.

Create a {level}-level learning roadmap in {language} for the course: '{course}'.
The user's goal is: '{goals}'.
if course is programming related and programming language is not mentioned then use python as programming language.
Return only valid JSON in the following format:
{format_instructions}
""")

        formatted_prompt = prompt_template.format(
            course=course, goals=goals, level=level, language=language,
            format_instructions=format_instructions
        )
        response = chat.invoke(formatted_prompt)
        raw_output = response.content.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("```json").strip("```")
        raw_output = raw_output.replace("{{", "{").replace("}}", "}")

        try:
            course_obj = parser.parse(raw_output)
            course_data = course_obj.model_dump()
        except Exception as e:
            return JsonResponse({'error': str(e), 'raw': raw_output}, status=500)

        course_data = generate_course_data_with_slugs(course_data)
        safe_email = request.user.email.replace("@", "_at_")
        filename = f"{safe_email}.json"
        if not filestorage.load_data(filename):
            filestorage.save_data(filename, {"courses": [{course+''+goals + '' + level + '' + language + '.json': {"score":0,"status":"pending"}}]})
        else:
            data111 = filestorage.load_data(filename)
            data111['courses'][course+''+goals + '' + level + '' + language + '.json'] =  {"score":0,"status":"pending"}
            filestorage.save_data(filename, data111 )
        filestorage.save_data(course+''+goals + '' + level + '' + language + '.json', course_data)

        # course_plan = CoursePlan.objects.create(
        #     user=request.user,
        #     course_name=course_data["course_name"],
        #     course_slug=course_data["course_slug"],
        #     level=course_data["level"],
        #     language=course_data["language"],
        #     goal=course_data["goal"],
        #     status=course_data.get("status", "In progress"),
        #     score=course_data.get("score", 0),
        #     max_score=course_data.get("max_score", len(course_data["modules"]))
        # )

        # for module in course_data["modules"]:
        #     mod = Module.objects.create(
        #         course_plan=course_plan,
        #         topic=module["topic"],
        #         topic_slug=module["topic_slug"],
        #         video_title=module.get("video_title"),
        #         video_link=module.get("video_link"),
        #         video_author=module.get("video_author"),
        #     )
        #     for sub in module["subtopics"]:
        #         Subtopic.objects.create(
        #             module=mod,
        #             name=sub["name"],
        #             slug=sub["slug"]
        #         )

        return JsonResponse({"success": True,"data":"new","error":"done"}, status=200)
        

    return render(request, 'submit_preferences.html')

def generate_course_data_with_slugs(course_data):
    course_data['course_slug'] = slugify(str(course_data["course_name"]))
    language = course_data.get('language', 'English')
    level = course_data.get('level', 'beginner')
    course_name = course_data.get('course_name', '')
    for module in course_data.get('modules', []):
        topic = module['topic']
        module['topic_slug'] = slugify(topic)
        
        # Fetch YouTube video info ONCE for this module
        video_data = fetch_youtube_video(course_name, topic, language, level)
        if(video_data):
            module['video_title'] = video_data['video_title']
            module['video_link'] = video_data['video_link']
            module['video_author'] = video_data['video_author']

        # Process subtopics
        processed_subtopics = []
        for sub in module.get('subtopics', []):
            sub_name = sub if isinstance(sub, str) else sub.get("name", "")
            sub_slug = slugify(sub_name)
            processed_subtopics.append({
                'name': sub_name,
                'slug': sub_slug
                # No video fields for subtopics
            })
        module['subtopics'] = processed_subtopics
    return course_data

def is_direct_youtube_video(url):
    return (
        url is not None and
        "youtube.com/watch?v=" in url or "youtu.be/" in url
    )

def fetch_youtube_video(course_name, topic_name, language="English", level="beginners"):
    params = {
        "engine": "google",
        "q": f"best {topic_name} tutorial for {course_name} comprehensive guide site:youtube.com in {language} for {level}",
        "api_key": os.environ.get('SERP_API', '')
    }
    try:
        res = requests.get("https://serpapi.com/search", params=params)
        data = res.json()
        if data.get("organic_results"):
            # Loop through each result and return first valid video link
            for result in data["organic_results"]:
                link = result.get("link")
                if is_direct_youtube_video(link):
                    title = result.get("title")
                    author = (
                        result.get('rich_snippet', {}).get('top', {}).get('channel', None)
                        or result.get('source', None)
                        or result.get('displayed_link', None)
                    )
                    return {
                        "video_link": link,
                        "video_title": title,
                        "video_author": author
                    }
            # If none found
            return {"video_link": None, "video_title": None, "video_author": None}

        
        
    except Exception:
        return {
            "video_link": None,
            "video_title": None,
            "video_author": None
        }


class ModuleOnSyllabusPydantic(BaseModel):
    topic: str
    subtopics: list[str]

class CoursePlanOnSyllabusPydantic(BaseModel):
    course_name: str
    modules: list[ModuleOnSyllabusPydantic]


from PIL import Image
import pytesseract
from django.core.files.uploadedfile import UploadedFile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile
@login_required(login_url='/login/')
def submit_syllabus_and_resources(request):
    if request.method == 'POST':
        parser = PydanticOutputParser(pydantic_object=CoursePlanOnSyllabusPydantic)
        format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        syllabus_file = request.FILES.get('syllabus')
        if not syllabus_file:
            return HttpResponse("No syllabus file uploaded", status=400)
        syllabus_file_type: UploadedFile = syllabus_file
        syllabus_type = syllabus_file_type.content_type
        if syllabus_type.startswith("image/"):
            try:
                syl_image = Image.open(syllabus_file)
                syl_text = pytesseract.image_to_string(syl_image)
            except Exception as e:
                return HttpResponse(f"Error processing syllabus image: {e}", status=400)

        elif syllabus_type == "application/pdf":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in syllabus_file.chunks():
                        tmp.write(chunk)
                    tmp.flush()
                    tmp_path = tmp.name
                syl_pdf = PyMuPDFLoader(tmp_path)
                syl_docs = syl_pdf.load()
                syl_text = "\n".join([doc.page_content for doc in syl_docs])
            except Exception as e:
                return HttpResponse(f"Error processing syllabus PDF: {e}", status=400)
            finally:
                # Ensure deletion after processing
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return HttpResponse("Invalid syllabus file type", status=400)
        
        # dup = get_duplicate_course(CoursePlanOnSyllabus.objects.filter(user=request.user))
        # if dup:
        #     return render(request, 'submit_syllabus_and_resources.html', {'error': f'Duplicate course found: {dup.course_name}'})
        
        resources = request.FILES.get('resources')
        res_text = ''
        if resources:
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in resources.chunks():
                        tmp.write(chunk)
                    tmp.flush()
                    tmp_path = tmp.name
                res_pdf = PyMuPDFLoader(tmp_path)
                res_docs = res_pdf.load()
                res_text = "\n".join([doc.page_content for doc in res_docs])
            finally:
                # Ensure deletion after processing
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        combined_text = f"SYLLABUS CONTENT:\n{syl_text}"
        if res_text:
            combined_text += f"\n\nRESOURCE MATERIALS:\n{res_text}"

        
        documents = splitter.create_documents([combined_text])
        vector_db = FAISS.from_documents(documents, embeddings)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieval_chain = custom_retrieval_chain(retriever, format_instructions)

        query = f"Analyze the syllabus content and extract a structured course plan"
        response = retrieval_chain.invoke({'input':query})
        raw_output = response["answer"].strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("```json").strip("```")
        raw_output = raw_output.replace("{{", "{").replace("}}", "}")
        course_plan = parser.parse(raw_output)
        course_data = course_plan.model_dump()
        course_data = generate_course_data_with_slugs_on_syllabus(course_data)
        course_plan = CoursePlanOnSyllabus.objects.create(
            user=request.user,
            syllabus=syllabus_file.read(),
            resources=resources.read() if resources else None,
            syllabus_text=syl_text,
            resources_text=res_text,
            course_name=course_data["course_name"],
            course_slug=course_data["course_slug"],
            status=course_data.get("status", "In progress"),
            score=course_data.get("score", 0),
            max_score=course_data.get("max_score", len(course_data["modules"]))
        )

        for module in course_data["modules"]:
            mod = ModuleOnSyllabus.objects.create(
                course_plan=course_plan,
                topic=module["topic"],
                topic_slug=module["topic_slug"],
            )
            for sub in module["subtopics"]:
                SubtopicOnSyllabus.objects.create(
                    module=mod,
                    name=sub["name"],
                    slug=sub["slug"]
                )

        return redirect(reverse('course_page_on_syllabus', kwargs={'course_slug': course_data['course_slug']}))

        
    return render(request, 'submit_syllabus_and_resources.html')



def custom_retrieval_chain(retriever, format_instructions):
    """
    Create a comprehensive retrieval chain for syllabus analysis
    """

    # System prompt for course plan extraction
    system_prompt = f"""
    You are an expert educational content analyzer. Your task is to analyze syllabus content and 
    extract a structured course plan with modules and subtopics.
    
    Guidelines:
    1. Identify the main course name from the syllabus
    2. Break down content into logical modules/chapters
    3. Extract specific subtopics for each module
    4. Ensure comprehensive coverage of all topics mentioned
    5. Organize content in a logical learning sequence
    
    Context from syllabus and resources:
    {{context}}

    Format your response exactly as specified in the format instructions below:
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        Please analyze the provided syllabus content and create a structured course plan.
        Focus on:
        1. Extracting the actual course name from the content
        2. Identifying main topics/modules
        3. Breaking down each module into specific subtopics
        4. Ensuring logical organization and flow

        Question: {input}
        """)
    ])

    retrieval_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm=chat, prompt=prompt)
    )
    return retrieval_chain

def generate_course_data_with_slugs_on_syllabus(course_data):
    course_data['course_slug'] = f"{slugify(str(course_data.get("course_name", "")))}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}"
    course_name = course_data.get('course_name', '')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for module in course_data.get('modules', []):
        topic = module['topic']
        # Module slug with timestamp
        module['topic_slug'] = f"{slugify(topic)}-{timestamp}"

        processed_subtopics = []
        for sub in module.get('subtopics', []):
            sub_name = sub if isinstance(sub, str) else sub.get("name", "")
            # Subtopic slug with timestamp
            sub_slug = f"{slugify(sub_name)}-{timestamp}"

            processed_subtopics.append({
                'name': sub_name,
                'slug': sub_slug
                # No video fields for subtopics
            })

        module['subtopics'] = processed_subtopics
    return course_data


#@login_required(login_url='/uped/login/')
# def course_page(request, course_slug):
#     try:
#         course = filestorage.load_data(course_slug + '.json')
#         user_courses = filestorage.load_data(request.user.email.replace("@", "_at_") + '.json')['courses'][0][course_slug+'.json']
#     except Http404:
#         course = get_object_or_404(CoursePlanOnSyllabus, course_slug=course_slug, user=request.user)
#     modules = course['modules']
#     #courses = data.get('courses', {})
#     return render(request, 'course.html', {
#         'course_name': course['course_name'],
#         'score': user_courses['score'],
#         'max_score': 10,
#         'status': user_courses['status'],
#         'topics': modules,
#         'course_slug': course_slug
#         })

@login_required(login_url='/login/')
def course_page(request, course_slug):
    try:
        # Load the course data
        course = filestorage.load_data(course_slug + '.json')

        # Load user’s progress file
        user_file = request.user.email.replace("@", "_at_") + '.json'
        user_data = filestorage.load_data(user_file)
        user_courses = user_data.get('courses', {})

        # Construct the key name (same format as you saved)
        course_key = course_slug + '.json'

        if course_key not in user_courses:
            # Handle case: user hasn’t started this course yet
            user_courses[course_key] = {'score': 0, 'status': 'Not Attempted'}

        # Extract course progress info
        progress = user_courses[course_key]
        modules = course['modules']

    except Http404:
        # If not found locally, fallback to DB
        course = get_object_or_404(CoursePlanOnSyllabus, course_slug=course_slug, user=request.user)
        modules = course['modules']
        progress = {'score': 0, 'status': 'Not Attempted'}

    # Render template
    return render(request, 'course.html', {
        'course_name': course['course_name'],
        'score': progress['score'],
        'max_score': 10,
        'status': progress['status'],
        'topics': modules,
        'course_slug': course_slug
    })

@login_required(login_url='/login/')
def topic_page(request, course_slug, topic_slug):
    try:
        course = filestorage.load_data(course_slug + '.json')
        user_courses = filestorage.load_data(
            request.user.email.replace("@", "_at_") + '.json'
        )['courses'][course_slug + '.json']
    except Http404:
        course = get_object_or_404(CoursePlanOnSyllabus, course_slug=course_slug, user=request.user)
    
    modules = course['modules']
    
    # ✅ Fix here: use string key 'topic_slug'
    topic = next((mod for mod in modules if mod['topic_slug'] == topic_slug), None)
    if not topic:
        return JsonResponse({'error': 'Topic not found'}, status=404)
    
    topic_data = {
        'topic': topic['topic'],
        'topic_slug': topic['topic_slug'],
        'video_title': topic.get('video_title'),
        'video_link': topic.get('video_link'),
        'video_author': topic.get('video_author'),
        'subtopics': [
            {
                'name': s['name'],
                'slug': s['slug'],
            } for s in topic['subtopics']
        ]
    }
    return JsonResponse({'topic': topic_data})

import json
import re
import requests
import base64
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

class GitHubRepoAnalyzer:
    """
    Analyzes GitHub repositories for project scoring
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize with optional GitHub token for higher rate limits
        Without token: 60 requests/hour
        With token: 5000 requests/hour
        """
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Educational-Platform-Bot'
        }
        
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
    
    def parse_github_url(self, github_url: str) -> tuple:
        """
        Parse GitHub URL to extract owner and repo name
        """
        # Handle different GitHub URL formats
        patterns = [
            r'github\.com/([^/]+)/([^/]+)/?$',
            r'github\.com/([^/]+)/([^/]+)\.git$',
            r'github\.com/([^/]+)/([^/]+)/.*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, github_url)
            if match:
                return match.group(1), match.group(2)
        
        raise ValueError("Invalid GitHub URL format")
    
    def get_repo_contents(self, owner: str, repo: str, path: str = '') -> List[Dict]:
        """
        Get repository contents from GitHub API
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repo contents: {e}")
            return []
    
    def get_file_content(self, owner: str, repo: str, file_path: str) -> str:
        """
        Get content of a specific file from GitHub
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{file_path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            file_data = response.json()
            
            if file_data.get('encoding') == 'base64':
                content = base64.b64decode(file_data['content']).decode('utf-8')
                return content
            else:
                return file_data.get('content', '')
        except Exception as e:
            print(f"Error fetching file {file_path}: {e}")
            return ""
    
    def analyze_repository_structure(self, owner: str, repo: str) -> Dict:
        """
        Analyze the overall structure of the repository
        """
        contents = self.get_repo_contents(owner, repo)
        
        structure = {
            'files': [],
            'directories': [],
            'readme_exists': False,
            'total_files': 0,
            'code_files': [],
            'documentation_files': [],
            'config_files': []
        }
        
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.php', '.rb', '.go', '.rs', '.ts', '.jsx', '.tsx', '.vue', '.swift', '.kt', '.dart', '.scala', '.r', '.m', '.sh'}
        doc_extensions = {'.md', '.txt', '.rst', '.doc', '.docx', '.pdf'}
        config_extensions = {'.json', '.yml', '.yaml', '.xml', '.ini', '.cfg', '.toml', '.env'}
        
        for item in contents:
            if item['type'] == 'file':
                structure['files'].append(item['name'])
                structure['total_files'] += 1
                
                # Check file extensions
                file_ext = '.' + item['name'].split('.')[-1].lower() if '.' in item['name'] else ''
                
                if file_ext in code_extensions:
                    structure['code_files'].append(item['name'])
                elif file_ext in doc_extensions:
                    structure['documentation_files'].append(item['name'])
                elif file_ext in config_extensions:
                    structure['config_files'].append(item['name'])
                
                # Check for README
                if item['name'].lower().startswith('readme'):
                    structure['readme_exists'] = True
            
            elif item['type'] == 'dir':
                structure['directories'].append(item['name'])
        
        return structure
    
    def get_repository_summary(self, github_url: str) -> Dict:
        """
        Get comprehensive repository analysis
        """
        try:
            owner, repo = self.parse_github_url(github_url)
            
            # Get repository info
            repo_url = f"{self.base_url}/repos/{owner}/{repo}"
            repo_response = requests.get(repo_url, headers=self.headers)
            repo_info = repo_response.json() if repo_response.ok else {}
            
            # Get repository structure
            structure = self.analyze_repository_structure(owner, repo)
            
            # Get README content if exists
            readme_content = ""
            if structure['readme_exists']:
                readme_files = [f for f in structure['files'] if f.lower().startswith('readme')]
                if readme_files:
                    readme_content = self.get_file_content(owner, repo, readme_files[0])
            
            # Get key code files content (limit to prevent API overuse)
            code_contents = {}
            for code_file in structure['code_files'][:5]:  # Limit to first 5 code files
                content = self.get_file_content(owner, repo, code_file)
                if content:
                    code_contents[code_file] = content[:2000]  # Limit content length
            
            return {
                'repo_info': {
                    'name': repo_info.get('name', repo),
                    'description': repo_info.get('description', ''),
                    'language': repo_info.get('language', 'Unknown'),
                    'created_at': repo_info.get('created_at', ''),
                    'updated_at': repo_info.get('updated_at', ''),
                    'size': repo_info.get('size', 0),
                    'stargazers_count': repo_info.get('stargazers_count', 0),
                    'forks_count': repo_info.get('forks_count', 0)
                },
                'structure': structure,
                'readme_content': readme_content,
                'code_contents': code_contents,
                'analysis_success': True
            }
            
        except Exception as e:
            print(f"Error analyzing repository: {e}")
            return {
                'analysis_success': False,
                'error': str(e)
            }

def ai_project_scorer_from_github(github_url: str, project_tasks: str, subtopic: str, course_name: str, topic: str) -> Dict:
    """
    AI-powered project scoring function that analyzes GitHub repository
    
    Args:
        github_url (str): GitHub repository URL
        project_tasks (str): Original project tasks generated
        subtopic (str): The subtopic name
        course_name (str): Course name
        topic (str): Topic name
    
    Returns:
        dict: Scoring results with score, feedback, and breakdown
    """
    
    # Initialize GitHub analyzer
    analyzer = GitHubRepoAnalyzer()  # Add GitHub token if available
    
    # Analyze the repository
    repo_analysis = analyzer.get_repository_summary(github_url)
    
    if not repo_analysis['analysis_success']:
        return {
            "overall_score": 0,
            "error": f"Failed to analyze repository: {repo_analysis.get('error', 'Unknown error')}",
            "detailed_feedback": "Could not access the GitHub repository. Please check the URL and ensure the repository is public."
        }

    
    # Create comprehensive prompt with repository data
    scoring_prompt = ChatPromptTemplate.from_template(
        """You are an expert code reviewer and educational assessor. Analyze the following GitHub repository submission for a student project.

EDUCATIONAL CONTEXT:
- Course: {course_name}
- Topic: {topic}
- Subtopic: {subtopic}
- Assigned Project Tasks: {project_tasks}

REPOSITORY ANALYSIS:
Repository Name: {repo_name}
Description: {repo_description}
Primary Language: {primary_language}
Last Updated: {last_updated}

REPOSITORY STRUCTURE:
- Total Files: {total_files}
- Code Files: {code_files}
- Documentation Files: {documentation_files}
- Has README: {has_readme}
- Directories: {directories}

README CONTENT:
{readme_content}

CODE SAMPLES:
{code_samples}

EVALUATION CRITERIA:
1. Task Completion (35%): How well does the repository address each required task?
2. Code Quality (25%): Is the code well-structured, readable, and follows best practices?
3. Documentation (20%): Quality of README, comments, and documentation
4. Repository Organization (10%): Proper file structure and organization
5. Innovation & Effort (10%): Evidence of additional effort, creativity, or going beyond requirements

Please provide a comprehensive evaluation in JSON format:
{{
    "overall_score": <number between 0-10>,
    "task_completion_score": <number between 0-10>,
    "code_quality_score": <number between 0-10>,
    "documentation_score": <number between 0-10>,
    "organization_score": <number between 0-10>,
    "innovation_score": <number between 0-10>,
    "detailed_feedback": "Comprehensive feedback explaining the scores and analysis",
    "strengths": ["strength1", "strength2", "strength3"],
    "areas_for_improvement": ["area1", "area2", "area3"],
    "specific_task_feedback": {{
        "task_1": "feedback for task 1 based on repository analysis",
        "task_2": "feedback for task 2 based on repository analysis",
        "task_3": "feedback for task 3 based on repository analysis"
    }},
    "code_review_comments": ["comment1", "comment2", "comment3"],
    "technical_observations": "Technical analysis of the implementation"
}}

Focus on educational value and provide constructive feedback that helps the student improve their programming and project management skills."""
    )
    
    try:
        # Prepare repository data for the prompt
        repo_info = repo_analysis['repo_info']
        structure = repo_analysis['structure']
        
        # Format code samples
        code_samples_text = ""
        for filename, content in repo_analysis['code_contents'].items():
            code_samples_text += f"\n--- {filename} ---\n{content}\n"
        
        if not code_samples_text:
            code_samples_text = "No code files found or accessible"
        
        # Get AI evaluation
        result = chat.invoke(scoring_prompt.format(
            course_name=course_name,
            topic=topic,
            subtopic=subtopic,
            project_tasks=project_tasks,
            repo_name=repo_info['name'],
            repo_description=repo_info['description'] or 'No description provided',
            primary_language=repo_info['language'],
            last_updated=repo_info['updated_at'],
            total_files=structure['total_files'],
            code_files=', '.join(structure['code_files']) if structure['code_files'] else 'None',
            documentation_files=', '.join(structure['documentation_files']) if structure['documentation_files'] else 'None',
            has_readme='Yes' if structure['readme_exists'] else 'No',
            directories=', '.join(structure['directories']) if structure['directories'] else 'None',
            readme_content=repo_analysis['readme_content'][:1000] if repo_analysis['readme_content'] else 'No README found',
            code_samples=code_samples_text
        ))
        
        # Parse the JSON response
        response_text = result.content
        
        # Clean the response if it contains code blocks
        if response_text.strip().startswith('```'):
            response_text = response_text.strip().strip('`')
            if "json" in response_text[:10].lower():
                response_text = response_text.split('\n', 1)[1]
        
        evaluation = json.loads(response_text)
        
        # Validate scores are within range
        score_fields = ['overall_score', 'task_completion_score', 'code_quality_score', 
                       'documentation_score', 'organization_score', 'innovation_score']
        
        for field in score_fields:
            if field in evaluation:
                evaluation[field] = max(0, min(10, evaluation[field]))
        
        # Add repository metadata to evaluation
        evaluation['repository_info'] = repo_info
        evaluation['repository_structure'] = structure
        evaluation['github_url'] = github_url
        
        return evaluation
        
    except Exception as e:
        print(f"AI Scoring Error: {e}")
        return {
            "overall_score": 0,
            "error": str(e),
            "detailed_feedback": f"Error processing repository analysis: {str(e)}",
            "github_url": github_url
        }

from django.views.decorators.csrf import csrf_exempt
from django.db.models import Sum
from bson import ObjectId
@csrf_exempt
def submit_quiz_score(request):
    """
    Expects JSON:
      {
        "score": 5,
        "course_id": "...",
        "topic": "...",
        "subtopic": "..."
      }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    data = json.loads(request.body)
    user = request.user
    score = int(data.get('score', 0))
    course = data.get('course_name')
    topic = data.get('topic')
    subtopic = data.get('subtopic')
    is_retake = data.get('is_retake', False)
    try:
        course_plan = CoursePlan.objects.get(course_name=course, user=request.user)
    except CoursePlan.DoesNotExist:
        return JsonResponse({"error": "Course not found"}, status=404)

    course_id = course_plan.id
    print(course_id)
    print(data)
    course_obj = Gen_Content.objects.get(
        user=user, course_id=course_id, topic_name=topic, subtopic_name=subtopic
    )
    
    if course_obj:
                # Update quiz score and status
                old_score = course_obj.quiz_score
                course_obj.quiz_score = score
                course_obj.quiz_status = 'Completed'
                course_obj.save()

                # Update course total score
                if is_retake:
                    # For retakes, adjust the course score difference
                    score_difference = score - old_score
                    course_obj.quiz_score += score_difference
                else:
                    # For first attempt, just add the score
                    course_obj.quiz_score += score

                course_obj.save()

    # Aggregate score summary
    all_rows = Gen_Content.objects.filter(user=user, course_id=course_id)
    quiz_total = all_rows.aggregate(q=Sum('quiz_score'))['q'] or 0
    proj_total = all_rows.aggregate(p=Sum('project_score'))['p'] or 0

    num_quiz = all_rows.count()
    num_proj = all_rows.exclude(project_score=0).count()
    total_possible = num_quiz * 5 + num_proj * 10
    total_score = quiz_total + proj_total

    
    course_plan.score = total_score
    course_plan.max_score = total_possible
    course_plan.save()
       

    return JsonResponse({
        "quiz_score": course_obj.quiz_score,
        "course_score": total_score,
        "total_max_score": total_possible
    })

@csrf_exempt
def submit_project_score(request):
    """
    Expects JSON:
      {
        "project_repo": "...",
        "course_id": "...",
        "topic": "...",
        "subtopic": "..."
      }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    data = json.loads(request.body)
    project_repo = data.get('project_repo')
    user = request.user
    course = data.get('course_name')
    topic = data.get('topic')
    subtopic = data.get('subtopic')
    is_retake = data.get('is_retake', False)
    try:
        course_plan = CoursePlan.objects.get(course_name=course, user=request.user)
    except CoursePlan.DoesNotExist:
        return JsonResponse({"error": "Course not found"}, status=404)
    course_id = course_plan.id
    print(data)

    course_obj = Gen_Content.objects.get(
        user=user, course_id=course_id, topic_name=topic, subtopic_name=subtopic
    )

    # Store old score for retakes
    old_project_score = course_obj.project_score
    
    # Update project repository
    course_obj.project_repo = project_repo
    course_obj.project_status = 'Completed'
    

    # AI score the repo
    evaluation = ai_project_scorer_from_github(
        github_url=project_repo,
        project_tasks=course_obj.project,
        subtopic=subtopic,
        course_name=course_plan.course_name,
        topic=topic
    )

    # Save
    if 'error' not in evaluation:
        new_project_score = min(int(evaluation['overall_score']), 10)
        course_obj.project_score = new_project_score
        course_obj.project_repo = project_repo
        course_obj.ai_feedback = json.dumps(evaluation)
        course_obj.project_status = 'Completed'
        course_obj.save()
        
        # Update course total score
        if is_retake:
            # For retakes, adjust the course score difference
            score_difference = new_project_score - old_project_score
            course_obj.project_score += score_difference
        else:
            # For first attempt, just add the score
            course_obj.project_score += new_project_score

        course_obj.save()
                

    # Update the total course score in Mongo
    all_rows = Gen_Content.objects.filter(user=user, course_id=course_id)
    quiz_total = all_rows.aggregate(q=Sum('quiz_score'))['q'] or 0
    proj_total = all_rows.aggregate(p=Sum('project_score'))['p'] or 0

    num_quiz = all_rows.count()
    num_proj = all_rows.exclude(project_score=0).count()
    total_possible = num_quiz * 5 + num_proj * 10
    total_score = quiz_total + proj_total

    
    course_plan.score = total_score
    course_plan.max_score = total_possible
    course_plan.save()

    return JsonResponse({
        "project_score": course_obj.project_score,
        "evaluation": evaluation,
        "course_score": total_score,
        "total_max_score": total_possible
    })

def convert_markdown_tables(text):
    """
    Convert markdown tables into HTML tables in the input text.
    """
    lines = text.split('\n')
    result_lines = []
    in_table = False
    table_lines = []

    def parse_table(table_lines):
        # Parse the header line and separator line first
        header_line = table_lines[0]
        separator_line = table_lines[1]
        data_lines = table_lines[2:]

        # Extract headers by splitting on '|', stripping spaces
        headers = [h.strip() for h in header_line.strip('| ').split('|')]
        # Validate separator line matches header count roughly (optional)

        # Build HTML table parts
        html_table = ['<table>']
        # Header row
        html_table.append('<thead><tr>')
        for header in headers:
            html_table.append(f'<th>{header}</th>')
        html_table.append('</tr></thead>')

        # Data rows
        html_table.append('<tbody>')
        for line in data_lines:
            if not line.strip():
                continue
            cols = [c.strip() for c in line.strip('| ').split('|')]
            html_table.append('<tr>')
            for col in cols:
                html_table.append(f'<td>{col}</td>')
            html_table.append('</tr>')
        html_table.append('</tbody></table>')

        return '\n'.join(html_table)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect if line is a potential table header (contains '|')
        # AND next line matches separator (like |---|---|---|)
        if (not in_table and '|' in line and i + 1 < len(lines) and
            re.match(r'^\s*\|?(\s*:?[-]+:?\s*\|)+\s*:?[-]+:?\s*\|?\s*$', lines[i+1])):
            in_table = True
            table_lines = [line, lines[i+1]]
            i += 2
            # Collect rest of table rows
            while i < len(lines) and (lines[i].strip().startswith('|') or lines[i].strip() == ''):
                if lines[i].strip() == '':
                    # Empty line ends the table
                    break
                table_lines.append(lines[i])
                i += 1
            # Parse collected table lines and append HTML to result
            html_table = parse_table(table_lines)
            result_lines.append(html_table)
            in_table = False
        else:
            result_lines.append(line)
            i +=1

    return '\n'.join(result_lines)

import re
import html
import uuid

def ai_response_to_html(text):
    """
    Convert AI-generated responses with hashtags, asterisks, markdown etc. to HTML,
    properly handling and preserving code blocks.
    """
    if not text:
        return ""

    # Step 1: Extract code blocks and replace with placeholders
    code_blocks = {}
    def code_block_replacer(match):
        code_content = match.group(1)
        placeholder = f"__CODE_BLOCK_{uuid.uuid4().hex}__"
        # Store escaped code inside placeholder to avoid mangling
        code_blocks[placeholder] = f"<pre><code>{html.escape(code_content)}</code></pre>"
        return placeholder

    text = re.sub(r'``````', code_block_replacer, text, flags=re.DOTALL)

    # Step 2: Escape HTML characters (except code blocks replaced)
    text = html.escape(text)

    # Step 3: Replace placeholders back to actual code blocks (unescaped)
    for placeholder, code_html in code_blocks.items():
        text = text.replace(html.escape(placeholder), code_html)  # placeholder was escaped in step 2

    # Step 4: Process headers (#, ##, ###)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Step 5: Bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)

    # Step 6: Italic (*text* or _text_)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!_)_([^_\n]+?)_(?!_)', r'<em>\1</em>', text)

    # Step 7: Inline code (`code`) — only outside code blocks, so safe now
    text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)

    # Step 8: Links [text](url)
    text = re.sub(r'\[([^\]]+?)\]\(([^)]+?)\)', r'<a href="\2">\1</a>', text)

    # Step 9: Unordered lists (- item or * item)
    lines = text.split('\n')
    in_list = False
    result_lines = []
    for line in lines:
        if re.match(r'^[\s]*[-*+]\s+', line):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            item_content = re.sub(r'^[\s]*[-*+]\s+', '', line)
            result_lines.append(f'<li>{item_content}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)
    if in_list:
        result_lines.append('</ul>')
    text = '\n'.join(result_lines)

    # Step 10: Numbered lists (1. item)
    lines = text.split('\n')
    in_ordered_list = False
    result_lines = []
    for line in lines:
        if re.match(r'^[\s]*\d+\.\s+', line):
            if not in_ordered_list:
                result_lines.append('<ol>')
                in_ordered_list = True
            item_content = re.sub(r'^[\s]*\d+\.\s+', '', line)
            result_lines.append(f'<li>{item_content}</li>')
        else:
            if in_ordered_list:
                result_lines.append('</ol>')
                in_ordered_list = False
            result_lines.append(line)
    if in_ordered_list:
        result_lines.append('</ol>')
    text = '\n'.join(result_lines)

    text = convert_markdown_tables(text)
    # Step 11: Blockquotes (> text)
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)

    # Step 12: Horizontal rules (--- or ***)
    text = re.sub(r'^---$', '<hr>', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\*\*$', '<hr>', text, flags=re.MULTILINE)

    # Step 13: Paragraphs and line breaks
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if paragraph.strip():
            # Don't wrap block elements in <p>
            if not re.match(r'^<(h[1-6]|ul|ol|blockquote|pre|hr)', paragraph.strip()):
                paragraph = paragraph.replace('\n', '<br>')
                paragraph = f'<p>{paragraph}</p>'
            else:
                paragraph = paragraph.replace('\n', '<br>')
            formatted_paragraphs.append(paragraph)

    text = '\n\n'.join(formatted_paragraphs)

    # Cleanup <br> around block elements
    text = re.sub(r'<br>\s*(</?(?:h[1-6]|ul|ol|li|blockquote|pre|hr)>)', r'\1', text)
    text = re.sub(r'(</?(?:h[1-6]|ul|ol|li|blockquote|pre|hr)>)\s*<br>', r'\1', text)

    return text.strip()


from pytube import YouTube

#@login_required(login_url='/uped/login/')
# def subtopic_page(request, course_slug, topic_slug, subtopic_slug):
#     try:
#         course = get_object_or_404(CoursePlan, course_slug=course_slug, user=request.user)
#     except Http404:
#         course = get_object_or_404(CoursePlanOnSyllabus, course_slug=course_slug, user=request.user)
#     modules = course.modules.prefetch_related('subtopics').all()
    
#     # Find the topic (module) based on topic_slug
#     topic = next((m for m in modules if m.topic_slug == topic_slug), None)
#     if not topic:
#         return render(request, "topic.html", {"error": "There is no such topic present in the course."})
    
#     # Find the subtopic based on subtopic_slug
#     subtopic = next((s for s in topic.subtopics.all() if s.slug == subtopic_slug), None)
#     if not subtopic:
#         return render(request, "subtopic.html", {"error": "There is no such subtopic present for this topic."})
    
#     # Build topics structure for navigation
#     topics = []
#     for mod in modules:
#         topics.append({
#             'topic': mod.topic,
#             'topic_slug': mod.topic_slug,
#             'subtopics': [
#                 {
#                     'name': s.name,
#                     'slug': s.slug,
#                 } for s in mod.subtopics.all()
#             ],
#         })
    
#     # Get video data from the module (topic) since that's where you store it
#     video_link = getattr(topic, 'video_link', None)  # Changed from subtopic to topic
#     video_title = getattr(topic, 'video_title', None)  # Added this
#     video_author = getattr(topic, 'video_author', None)  # Added this

#     # Extract video ID for YouTube embed
#     video_id = ''
#     if video_link:
#         try:
#             video_id = YouTube(video_link).video_id
#         except Exception:
#             video_id = ''
    
#     # Check if content already exists for this subtopic
#     query_set = Gen_Content.objects.filter(
#         course_id=str(course.id),
#         topic_name=topic.topic,
#         subtopic_name=subtopic.name,
#         user=request.user
#     )
    
#     if query_set.exists():
#         course_obj = query_set.first()
#         notes, quizzes, project = course_obj.notes, course_obj.quizzes, course_obj.project
#     else:
#         if getattr(course, 'resources_text', None):
#             notes, quizzes, project = note_quiz_project_on_syllabus(course.course_name, topic.topic, subtopic.name, course.resources_text)
#         else:
#             notes, quizzes, project = note_quiz_project_gen(course.course_name, topic.topic, subtopic.name)

#         # Create new Gen_Content object
#         course_obj = Gen_Content(
#             user=request.user,
#             course_id=str(course.id),
#             topic_name=topic.topic,
#             subtopic_name=subtopic.name,
#             notes=notes,
#             quizzes=quizzes,
#             project=project,
#             quiz_score=0,
#             project_score=0,
#             quiz_status='Pending',
#             project_status='Pending',
#             project_repo=None,
#             ai_feedback=None
#         )
#         course_obj.save()

#     return render(request, "subtopic.html", {
#         'notes': notes,
#         'quiz': quizzes,
#         'project': project,
#         'video_link': video_link,
#         'video_id': video_id,
#         'video_title': video_title,  # Added this
#         'course_name': course.course_name,
#         'topic': topic.topic,
#         'subtopic': subtopic.name,
#         'topics': topics,
#         'subtopic_slug': subtopic.slug,
#         'topic_slug': topic.topic_slug,  # Added this for navigation
#         'course_slug': course.course_slug,
#         'quiz_status': course_obj.quiz_status,
#         'project_status': course_obj.project_status,
#         'quiz_score': course_obj.quiz_score,
#         'project_score': course_obj.project_score,
#         'project_repo': course_obj.project_repo,
#         'ai_feedback': ai_response_to_html(course_obj.ai_feedback) if course_obj.ai_feedback else None,
#         'author': video_author,  # Changed from subtopic.video_author to video_author
#     })

@login_required(login_url='/login/')
def subtopic_page(request, course_slug, topic_slug, subtopic_slug):
    try:
        # Load course data from JSON file
        course = filestorage.load_data(course_slug + '.json')
        user_courses = filestorage.load_data(request.user.email.replace("@", "_at_") + '.json')['courses'][course_slug+'.json']
    except FileNotFoundError:
        return JsonResponse({'error': 'Course data not found'}, status=404)

    modules = course.get('modules', [])

    # Find the topic
    topic = next((m for m in modules if m.get('topic_slug') == topic_slug), None)
    if not topic:
        return JsonResponse({'error': 'Topic not found'}, status=404)

    # Find the subtopic
    subtopic = next((s for s in topic.get('subtopics', []) if s.get('slug') == subtopic_slug), None)
    if not subtopic:
        return JsonResponse({'error': 'Subtopic not found'}, status=404)

    # Get video info from topic
    video_link = topic.get('video_link')
    video_title = topic.get('video_title')
    video_author = topic.get('video_author')

    # Extract video ID for YouTube embed
    video_id = ''
    if video_link:
        try:
            video_id = YouTube(video_link).video_id
        except Exception:
            video_id = ''

    # Prepare topic-subtopic navigation
    topics_navigation = []
    for mod in modules:
        topics_navigation.append({
            'topic': mod.get('topic'),
            'topic_slug': mod.get('topic_slug'),
            'subtopics': [{'name': s.get('name'), 'slug': s.get('slug')} for s in mod.get('subtopics', [])]
        })

    # For notes/quizzes/project, you can still call your existing functions
    notes, quizzes, project = note_quiz_project_gen(course.get('course_name'), topic.get('topic'), subtopic.get('name'))

    return JsonResponse({
        'course_name': course.get('course_name'),
        'course_slug': course_slug,
        'topic': {
            'name': topic.get('topic'),
            'slug': topic.get('topic_slug'),
            'video_title': video_title,
            'video_link': video_link,
            'video_id': video_id,
            'video_author': video_author,
            'subtopics': topic.get('subtopics', [])
        },
        'subtopic': {
            'name': subtopic.get('name'),
            'slug': subtopic.get('slug'),
            'notes': notes,
            'quizzes': quizzes,
            'project': project
        },
        'topics_navigation': topics_navigation
    })



from youtube_transcript_api import YouTubeTranscriptApi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from markdown import markdown
# def notes_quizz_and_project_gen_from_yt(video_link, subtopic):
#     yt = YouTube(video_link)
#     video_id = yt.video_id
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)
#     full_text = " ".join([t['text'] for t in transcript])

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = splitter.create_documents([full_text])

#     # Make sure 'chat' and 'embeddings' are defined globally or pass as arguments
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY"))
#     db = Chroma.from_documents(docs, embeddings)
#     retriever = db.as_retriever()

#     # NOTES
#     notes_prompt = ChatPromptTemplate.from_template(
#         "You are an expert note-taker. Summarize the following transcript as detailed notes on the subtopic '{subtopic}':\n{context}.\nOnly use transcript segments relevant to the subtopic '{subtopic}'."
#     )
#     notes_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(chat, notes_prompt))
#     notes_result = notes_chain.invoke({"input": subtopic, "subtopic": subtopic})
#     notes_output = ai_response_to_html(notes_result['answer'] if isinstance(notes_result, dict) else str(notes_result))

#     # QUIZ
#     quiz_prompt = ChatPromptTemplate.from_template(
#         """You are a MCQs generator. Create 5 MCQ with their options and an answer relevant to the subtopic '{subtopic}' based on this transcript:\n{context}.\nOnly use transcript segments relevant to the subtopic '{subtopic}'.
#         Output as a JSON array where each item is:
#         {{
#           "question": "...",
#           "options": ["...", "...", "...", "..."],
#           "answer": "..."
#         }}"""
#     )
#     quiz_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(chat, quiz_prompt))
#     quiz_result = quiz_chain.invoke({"input": subtopic, "subtopic": subtopic})
#     quiz_output = quiz_result['answer'] if isinstance(quiz_result, dict) else str(quiz_result)

#     try:
#         if quiz_output.strip().startswith('```'):
#             quiz_output = quiz_output.strip().strip('`')
#             if "json" in quiz_output[:10].lower():
#                 quiz_output = quiz_output.split('\n', 1)[1]
#         quiz_json = json.loads(quiz_output)
#     except Exception as e:
#         print("Quiz to JSON parse error: ", e)
#         quiz_json = []

#     # PROJECT
#     project_prompt = ChatPromptTemplate.from_template(
#         "You are a project generator. Create a project with 3 tasks based on the subtopic '{subtopic}' and this transcript:\n{context}.\nOnly use transcript segments relevant to the subtopic '{subtopic}'."

#     )
#     project_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(chat, project_prompt))
#     project_result = project_chain.invoke({"input": subtopic, "subtopic": subtopic})
#     project_output = ai_response_to_html(project_result['answer'] if isinstance(project_result, dict) else str(project_result))

#     return notes_output, quiz_json, project_output
def note_quiz_project_on_syllabus(course_name, topic, subtopic, resources):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = splitter.create_documents([resources])
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    notes_prompt = ChatPromptTemplate.from_template("""
    You are a note generator. Your task is to generate detailed notes on `{subtopic}`, 
    which is a subtopic of `{topic}` in `{course_name}`.
    
    Use the context from the syllabus/resources to ensure accuracy:
    {context}
    
    Output only the notes. Do not add "next steps" or extra commentary.
    """)
    
    notes_chain = create_stuff_documents_chain(llm=chat, prompt=notes_prompt)
    notes_retrieval_chain = create_retrieval_chain(retriever, notes_chain)
    
    notes = notes_retrieval_chain.invoke({
        "input": f"Generate notes for {subtopic} in {topic} ({course_name})",
        "course_name": course_name,
    "topic": topic,
    "subtopic": subtopic
    })
    notes_output = ai_response_to_html(notes["answer"])

    # QUIZZES
    quiz_prompt = PromptTemplate.from_template("""
    You are a MCQ generator. Based on the notes:
    `{notes}`
    
    Generate 5 MCQs with options and an answer for `{subtopic}` (subtopic of `{topic}` in `{course_name}`).
    Do not add any extra commentary. 

    Output must be ONLY a valid JSON array where each item is:
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "answer": "..."
    }}
    """)
    
    input_data = {
        "notes": notes["answer"],
        "subtopic": subtopic,
        "course_name": course_name,
        "topic": topic
    }
    quiz_chain = quiz_prompt | chat
    quizzes = quiz_chain.invoke(input_data)
    quiz_output = quizzes.content

    try:
        if quiz_output.strip().startswith('```'):
            quiz_output = quiz_output.strip().strip('`')
            if "json" in quiz_output[:10].lower():
                quiz_output = quiz_output.split('\n', 1)[1]
        quiz_json = json.loads(quiz_output)
    except Exception as e:
        print("Quiz to JSON parse error (on_syllabus):", e)
        quiz_json = []

    # PROJECT
    project_prompt = PromptTemplate.from_template("""
    You are a project generator. Based on the notes:
    `{notes}`

    Create a simple project idea (not a solution!) with small tasks and hints 
    that are achievable with the given notes only. Keep it scoped to `{subtopic}` in `{topic}` for `{course_name}`. 
    Don't add extra content.
    """)
    
    project_chain = project_prompt | chat
    project = project_chain.invoke(input_data)
    project_output = ai_response_to_html(project.content)

    return notes_output, quiz_json, project_output


def note_quiz_project_gen(course_name, topic, subtopic):
    templ123 = """You are a professional course content creator. Generate clear, easy-to-understand notes on the subtopic “{subtopic}”, which is part of the topic “{topic}” in the course “{course_name}”.

    Write the notes in simple, student-friendly language, using short paragraphs, bullet points, and real-life examples wherever possible.
    1. Use headings and subheadings to organize the content.
    2. wrap code snippets in <pre><code> blocks.
    3. Use bold and italics to highlight important terms.


    Don’t include extra sections like “next steps,” “summary,” or unrelated text — only the main notes for the subtopic."""
    # note_prompt = PromptTemplate.from_template("You are a note generator. You have to generate notes on `{subtopic}` which is a subtopic of `{topic}` in `{course_name}`. Don't generate `next steps` and other text except what i have asked for.")
    note_prompt = PromptTemplate.from_template(templ123)
    notes_chain = note_prompt | chat
    notes = notes_chain.invoke({'subtopic': subtopic, 'course_name': course_name, 'topic': topic})
    notes_output = ai_response_to_html(notes.content)
    quiz_prompt = PromptTemplate.from_template("""You are a MCQs generator. Based on \n `{notes}` \n You have to generate 5 MCQs with their options and an answer on `{subtopic}` which is a subtopic of `{topic}` in `{course_name}`. Don't generate other text except what i have asked for.
                          Output as a JSON array where each item is:
        {{
          "question": "...",
          "options": ["...", "...", "...", "..."],
          "answer": "..."
        }}""")
    input = {
        "notes": notes.content,
    "subtopic": subtopic,
    "course_name": course_name,
    "topic": topic
    }
    quiz_chain = quiz_prompt | chat
    quizzes = quiz_chain.invoke(input)
    quiz_output = quizzes.content
    print(quiz_output)
    try:
        if quiz_output.strip().startswith('```'):
            quiz_output = quiz_output.strip().strip('`')
            if "json" in quiz_output[:10].lower():
                quiz_output = quiz_output.split('\n', 1)[1]
        quiz_json = json.loads(quiz_output)
    except Exception as e:
        print("Quiz to JSON parse error: ", e)
        quiz_json = []
    project_prompt = PromptTemplate.from_template("You are a project generator. Based on \n`{notes}`\n the above notes you have to generate a project without answering it with a simple tasks including the things present in the notes not more than that as the user don't have knowledge about it and small hints for it based on `{subtopic}` which is a subtopic of `{topic}` in `{course_name}`. Don't generate other text except what i have asked for.")
    
    project_chain = project_prompt | chat
    project = project_chain.invoke(input)
    project_output = ai_response_to_html(project.content)
    return notes_output, quiz_json, project_output


from django.template.loader import get_template

@login_required(login_url='/login/')
def certificate_gen(request, course_slug):
    course = get_object_or_404(CoursePlan, course_slug=course_slug, user=request.user)
    completion_date = course.completion_date or datetime.date.today()
    context = {
        'course_name': course.course_name,
        'username': request.user.name,
        'completion_date': completion_date.strftime('%Y-%m-%d'),
    }
    certification_id = f"PEP{completion_date.strftime('%Y-%m-%d')}{random.randint(10,99)}"
    context['cert_id'] = certification_id
    return render(request, 'certificate.html', context)

from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.config import RunnableConfig
from typing import List, Dict, Any

session_store: Dict[str, ChatMessageHistory] = {}
def get_session_history_for_langchain(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

@csrf_exempt
@login_required(login_url='/login/')
def chatbot(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data['message']
        session_id = request.session.session_key
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                    You are an expert programming mentor. Answer only programming-related questions. 
                    If asked anything else, respond with 'I'm sorry, I can only assist with programming topics.'
                """),
                MessagesPlaceholder(variable_name="history"), 
                ("human", "{user_message}"), 
            ]
        )
        chain = (
            RunnablePassthrough.assign(
                # Use RunnableLambda with `RunnableConfig` to correctly access session_id
                # `config` is automatically available within RunnableLambda when wrapped
                # by RunnableWithMessageHistory
                history=RunnableLambda(
                    lambda x, config: get_session_history_for_langchain(config["configurable"]["session_id"]).messages
                ).with_config(run_name="get_history_from_session") # Adding a run_name for clarity in tracing
            )
            | prompt
            | chat
        )
        runnable_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history_for_langchain,
            input_messages_key="user_message",
            history_messages_key="history",
        )
        response = runnable_with_history.invoke({'user_message': user_message}, config = {"configurable":{"session_id": session_id}})
        response_text = response.content
        return JsonResponse({'message': response_text})
    return render(request, 'chatbot.html')

#subtopic Chat

@csrf_exempt
@login_required(login_url='/login/')
def subtopic_chat(request, course_slug, topic_slug, subtopic_slug):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message')
        session_id = request.session.session_key
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                    You are a friendly and knowledgeable course mentor for an e-learning platform named as UpEd adn your name is UpEd Mentor.

The student is currently studying the subtopic "{subtopic_slug}" under the topic "{topic_slug}" in the course "{course_slug}".

Your goal:
- Act like a helpful teacher who explains clearly and patiently.
- Answer only questions related to the current subtopic.
- Use simple language, real-life analogies, and small examples with story or code snippets when needed.
- Encourage and motivate the student to keep learning.
- Break down complex ideas into small, understandable parts.
- If the question goes beyond this subtopic, gently say:
  "Let's stay focused on the current subtopic: {subtopic_slug}. We can explore other topics once you complete this section."
- use HTML formatting like <br>,<b>, <i>, <ul>, <ol>, <li>, <pre><code> for code blocks, etc. to make your answers more engaging and easier to read.

Guidelines:
- Do **not** show next steps or unrelated topics.
- Do **not** restate the entire subtopic content.
- Keep responses **educational, conversational, and encouraging** — as if you're personally guiding the learner.

if user is asking for any actions like next lesson, previous lesson, open quiz, start project, etc. respond in the below JSON format only:
{{
 "reply": "<short friendly message to user>",
 "actions": [
   {{"type":"NAVIGATE","target":"lesson","params":{{"lessonId":"lesson_23"}}}},
   ...
 ]
}}
Examples of user messages that require JSON action replies:
- User: "next" or "continue" → {{"actions":[{{"type":"NAVIGATE","target":"next_lesson"}}]}}
- User: "previous" → {{"actions":[{{"type":"NAVIGATE","target":"previous_lesson"}}]}}
- User: "open quiz" → {{"actions":[{{"type":"OPEN_MODAL","target":"quiz"}}]}}
- User: "start project" → {{"actions":[{{"type":"NAVIGATE","target":"project"}}]}}

Make sure JSON is valid. If the user intent is ambiguous, return "actions":[] and use reply to ask a clarifying question. if there will any issue in completing that task next automatic message from system will tell u that then respond accordingly.


Example tone:
“Good question! Imagine you’re trying to fit a straight line through a scatter of points — that’s exactly what linear regression does. Let’s break it down step by step…”


                """),
                MessagesPlaceholder(variable_name="history"), 
                ("human", "{user_message}"), 
            ]
        )
        chain = (
            RunnablePassthrough.assign(
                # Use RunnableLambda with `RunnableConfig` to correctly access session_id
                # `config` is automatically available within RunnableLambda when wrapped
                # by RunnableWithMessageHistory
                history=RunnableLambda(
                    lambda x, config: get_session_history_for_langchain(config["configurable"]["session_id"]).messages
                ).with_config(run_name="get_history_from_session") # Adding a run_name for clarity in tracing
            )
            | prompt
            | chat
        )
        runnable_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history_for_langchain,
            input_messages_key="user_message",
            history_messages_key="history",
        )
        response = runnable_with_history.invoke({'user_message': user_message,'course_slug':course_slug,'topic_slug':topic_slug,'subtopic_slug':subtopic_slug,'reply':'doing your task'}, config = {"configurable":{"session_id": session_id}})
        response_text = response.content
        return JsonResponse({'message': response_text})
    return JsonResponse({'message': 'error baby'})


from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import PyPDF2
from django.db.models import Avg
import io
from .models import InterviewSession, InterviewQuestion

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage


@login_required(login_url='/login/')
def interview(request):
    """Main interview application page"""
    return render(request, 'interview.html')

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            for chunk in pdf_file.chunks():
                tmp.write(chunk)
            tmp.flush()
            tmp_path = tmp.name
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@csrf_exempt
@require_http_methods(["POST"])
@login_required(login_url='/login/')
def start_interview(request):
    """Initialize new interview session"""
    try:
        if 'resume_pdf' in request.FILES:
            pdf_file = request.FILES['resume_pdf']

            # Read file data as bytes
            pdf_bytes = pdf_file.read()
            pdf_name = pdf_file.name

            # Extract text from PDF (using the file object or its bytes)
            pdf_file.seek(0)
            user_content = extract_text_from_pdf(pdf_file)
            input_type = 'PDF'
            user = request.user

            # Create interview session and store PDF in DB
            session = InterviewSession.objects.create(
                user=user,
                user_input_type=input_type,
                user_input_content=user_content,
                resume_pdf_data=pdf_bytes,
                resume_pdf_name=pdf_name,
                current_question_index=0,
            )
        
            
        else:
            # Get skills from form data
            user_content = request.POST.get('domain_skills', '').strip()
            if not user_content:
                return JsonResponse({'error': 'Please provide either a resume PDF or skills/domain information.'}, status=400)
            
            file_path = None
            input_type = 'TEXT'
            # Create interview session
            session = InterviewSession.objects.create(
                user=request.user,
                user_input_type=input_type,
                user_input_content=user_content,
                current_question_index=0
            )
        
        # Generate first question using LLM
        first_question = generate_first_question(user_content, input_type)
        
        # Create first question record
        question = InterviewQuestion.objects.create(
            session=session,
            question_number=1,
            question_text=first_question,
            timestamp_asked=timezone.now()
        )
        
        return JsonResponse({
            'session_id': str(session.session_id),
            'question': first_question,
            'question_number': 1,
            'total_questions': 7
        })
        
    except Exception as e:
        print(e)
        log_exception(e)
        return JsonResponse({'error': f'Failed to start interview: {str(e)}'}, status=500)
    
def _parse_fallback_response(response_text, user_response):
    """Parse response when JSON parsing fails"""
    # Extract score if possible
    score_match = re.search(r'score[\'"]?\s*:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else 7.0
    
    return {
        'feedback': response_text[:500] + "..." if len(response_text) > 500 else response_text,
        'score': min(10.0, max(1.0, score)),
        'needs_clarification': False,
        'clarification_prompt': None
    }

def _generate_fallback_evaluation(user_response, question_number):
    """Generate fallback evaluation when API fails"""
    if not user_response.strip():
        return {
            'feedback': "No response provided. Please provide an answer to continue.",
            'score': 1.0,
            'needs_clarification': True,
            'clarification_prompt': "Could you please provide an answer to the question?"
        }
    
    feedback_templates = [
        "Thank you for your response. You've provided relevant information about your background. Consider adding more specific examples and elaborating on your achievements.",
        "Good answer! You've covered the key points. Try to provide more concrete examples and quantify your achievements where possible.",
        "Your response shows good self-awareness. Consider structuring your answer with specific examples that demonstrate your skills and experience.",
    ]
    
    return {
        'feedback': random.choice(feedback_templates),
        'score': random.uniform(6.5, 8.5),
        'needs_clarification': False,
        'clarification_prompt': None
    }

def generate_first_question(user_content, input_type):
    """Generate the first interview question based on user input"""
    if chat is None:
        return "Tell me about yourself and your professional background."
    
    try:
        content_type = "resume" if input_type == "PDF" else "skills and domain information"
        
        system_prompt = f"""You are an experienced technical interviewer conducting a mock interview. 
        Based on the candidate's {content_type}, generate the first of 7 interview questions.
        
        Guidelines:
        - Create questions that are relevant to their background
        - Mix technical, behavioral, and situational questions across the interview
        - Start with a good opening question
        - Only return the question text, no additional formatting
        - Keep questions clear and professional
        
        Candidate's {content_type}:
        {user_content}
        
        Generate the first interview question:"""
        
        response = chat.invoke(system_prompt)
        return response.content.strip()
        
    except Exception as e:
        # Fallback question
        print(f"Error generating first question: {e}")

@csrf_exempt
@require_http_methods(["POST"])
@login_required(login_url='/login/')
def submit_answer(request):
    """Process user's answer and provide feedback"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        user_response = data.get('user_response', '').strip()
        is_timeout = data.get('is_timeout', False)
        
        # Get session and current question
        session = InterviewSession.objects.get(session_id=session_id)
        current_question = session.questions.filter(
            question_number=session.current_question_index + 1
        ).first()
        
        if not current_question:
            return JsonResponse({'error': 'Question not found'}, status=400)
        
        # Handle timeout case
        if is_timeout and not user_response:
            user_response = "[No response - timeout]"
        
        # Update question with user's answer
        current_question.user_answer = user_response
        current_question.timestamp_answered = timezone.now()
        current_question.save()
        
        # Get AI feedback and evaluation
        feedback_data = evaluate_response(
            current_question.question_text,
            user_response,
            session.user_input_content,
            session.current_question_index + 1,
            7
        )
        
        # Update question with AI feedback
        current_question.ai_feedback = feedback_data['feedback']
        current_question.score = feedback_data['score']
        current_question.needs_clarification = feedback_data['needs_clarification']
        current_question.clarification_prompt = feedback_data.get('clarification_prompt', '')
        current_question.save()
        
        response_data = {
            'feedback': feedback_data['feedback'],
            'score': feedback_data['score'],
            'needs_clarification': feedback_data['needs_clarification']
        }
        
        # If clarification needed, return clarification prompt
        if feedback_data['needs_clarification']:
            response_data['clarification_prompt'] = feedback_data['clarification_prompt']
            response_data['question_number'] = session.current_question_index + 1
            response_data['total_questions'] = 7
            return JsonResponse(response_data)
        
        # Move to next question or complete interview
        session.current_question_index += 1
        
        if session.current_question_index >= 7:
            # Interview completed
            session.is_completed = True
            # Calculate total score
            total_score = session.questions.aggregate(
                avg_score=Avg('score')
            )['avg_score'] or 0
            session.total_score = total_score
            session.save()
            
            response_data.update({
                'is_interview_complete': True,
                'total_score': round(total_score, 1),
                'session_summary': generate_interview_summary(session)
            })
        else:
            # Generate next question
            next_question_text = generate_next_question(
                session.user_input_content,
                session.current_question_index + 1,
                session
            )
            
            # Create next question record
            next_question = InterviewQuestion.objects.create(
                session=session,
                question_number=session.current_question_index + 1,
                question_text=next_question_text,
                timestamp_asked=timezone.now()
            )
            
            response_data.update({
                'next_question': next_question_text,
                'question_number': session.current_question_index + 1,
                'total_questions': 7,
                'is_interview_complete': False
            })
        
        session.save()
        return JsonResponse(response_data)
        
    except InterviewSession.DoesNotExist:
        return JsonResponse({'error': 'Interview session not found'}, status=404)
    except Exception as e:
        print(e)
        return JsonResponse({'error': f'Failed to process answer: {str(e)}'}, status=500)

def evaluate_response(question, user_response, background, question_number, total_questions):
    """Evaluate user's response using Gemini"""
    try:
        system_prompt = """You are an expert interviewer evaluating a candidate's response.
        Provide constructive feedback and scoring."""
        
        evaluation_prompt = f"""Evaluate this interview response:

Question: {question}
Candidate's Response: {user_response}
Candidate's Background: {background}
Question Number: {question_number}/{total_questions}

Please evaluate this response and provide:
1. Detailed feedback (both strengths and areas for improvement)
2. A score from 1-10
3. Whether the response needs clarification (true/false)
4. If clarification is needed, provide a follow-up question

Format your response as JSON:
{{
    "feedback": "detailed feedback here",
    "score": 8.5,
    "needs_clarification": false,
    "clarification_prompt": "follow-up question if needed"
}}"""

        print(f'DEBUG: Evaluating response: {evaluation_prompt[:200]}...')
        
        # Use both SystemMessage and HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=evaluation_prompt)
        ]
        
        response = chat.invoke(messages)
        
        print(f"DEBUG: AI Response: {response.content}")
        
        # Parse JSON response
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return _parse_fallback_response(response.content, user_response)
            
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return _generate_fallback_evaluation(user_response, question_number)

def generate_next_question(user_content, question_number, session):
    """Generate the next interview question"""
    if chat is None:
        fallback_questions = [
            "What are your greatest strengths and how do they apply to this field?",
            "Describe a challenging project you worked on and how you handled it.",
            "How do you stay updated with the latest trends in your domain?",
            "Tell me about a time when you had to learn something new quickly.",
            "What are your career goals for the next 3-5 years?",
            "How do you handle working under pressure or tight deadlines?",
            "Do you have any questions about our company or this role?"
        ]
        return fallback_questions[min(question_number - 1, len(fallback_questions) - 1)]
    
    try:
        # Get previous questions for context
        previous_questions = list(session.questions.filter(
            question_number__lt=question_number
        ).values('question_text', 'user_answer'))
        
        context = "\n".join([
            f"Q{i+1}: {q['question_text']}\nA: {q['user_answer']}\n"
            for i, q in enumerate(previous_questions)
        ])
        
        system_prompt = f"""You are conducting a technical interview. This is question {question_number} of 7.
        
        Candidate's background: {user_content}
        
        Previous questions and answers:
        {context}
        
        Generate the next question that:
        - Builds on previous responses where relevant
        - Covers different aspects (technical, behavioral, situational)
        - Is appropriate for question {question_number}/7
        - Only return the question text, no formatting
        
        Generate question {question_number}:"""
        
        response = chat.invoke(system_prompt)
        return response.content.strip()
        
    except Exception as e:
        print(e)

def generate_interview_summary(session):
    """Generate a summary of the completed interview"""
    questions = session.questions.all().order_by('question_number')
    avg_score = session.total_score or 0
    
    summary = f"""Interview Summary:
    
Total Score: {avg_score:.1f}/10
Questions Answered: {questions.count()}/7

Performance Overview:
{"Excellent" if avg_score >= 8 else "Good" if avg_score >= 6 else "Needs Improvement"}

Key Strengths and Areas for Improvement will be detailed in your downloadable report.
"""
    return summary



@csrf_exempt
@require_http_methods(["GET"])
def download_results(request, session_id):
    """Download interview results as a formatted text file"""
    try:
        session = InterviewSession.objects.get(session_id=session_id)
        questions = session.questions.all().order_by('question_number')
        
        # Generate formatted report
        report = f"""AI INTERVIEW PREPARATION REPORT
{'='*50}

Session ID: {session.session_id}
Date: {session.timestamp.strftime('%B %d, %Y at %I:%M %p')}
Input Type: {session.get_user_input_type_display()}
Overall Score: {session.total_score:.1f}/10

CANDIDATE BACKGROUND:
{session.user_input_content}

{'='*50}
INTERVIEW QUESTIONS AND RESPONSES
{'='*50}

"""
        
        for question in questions:
            report += f"""
QUESTION {question.question_number}/7:
{question.question_text}

YOUR RESPONSE:
{question.user_answer or '[No response provided]'}

AI FEEDBACK:
{question.ai_feedback or '[No feedback available]'}

SCORE: {question.score or 'N/A'}/10

{'-'*50}
"""
        
        report += f"""

INTERVIEW SUMMARY:
{generate_interview_summary(session)}

Generated on: {timezone.now().strftime('%B %d, %Y at %I:%M %p')}
"""
        
        # Create response
        response = HttpResponse(report, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="interview_results_{session_id}.txt"'
        return response
        
    except InterviewSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to generate report: {str(e)}'}, status=500)


# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from io import BytesIO

# @csrf_exempt
# @require_http_methods(["GET"])
# def download_results(request, session_id):
#     """Download interview results as a PDF file"""
#     try:
#         session = InterviewSession.objects.get(session_id=session_id)
#         questions = session.questions.all().order_by('question_number')

#         # Create PDF buffer
#         buffer = BytesIO()
#         p = canvas.Canvas(buffer, pagesize=letter)
#         width, height = letter

#         # Title
#         p.setFont("Helvetica-Bold", 14)
#         p.drawString(50, height - 50, "AI INTERVIEW PREPARATION REPORT")
#         p.setFont("Helvetica", 10)

#         # Session Details
#         p.drawString(50, height - 80, f"Session ID: {session.session_id}")
#         p.drawString(50, height - 95, f"Date: {session.timestamp.strftime('%B %d, %Y at %I:%M %p')}")
#         p.drawString(50, height - 110, f"Input Type: {session.get_user_input_type_display()}")
#         score = session.total_score if session.total_score is not None else 0.0
#         p.drawString(50, height - 125, f"Overall Score: {score:.1f}/10")


#         # Candidate Background
#         p.setFont("Helvetica-Bold", 12)
#         p.drawString(50, height - 150, "CANDIDATE BACKGROUND:")
#         p.setFont("Helvetica", 10)
#         text_object = p.beginText(50, height - 165)
#         text_object.setFont("Helvetica", 10)
#         for line in session.user_input_content.splitlines():
#             text_object.textLine(line)
#         p.drawText(text_object)

#         # Questions and Answers
#         y = height - 200
#         for q in questions:
#             if y < 100:
#                 p.showPage()
#                 y = height - 50

#             p.setFont("Helvetica-Bold", 11)
#             p.drawString(50, y, f"QUESTION {q.question_number}/7: {q.question_text}")
#             y -= 14
#             p.setFont("Helvetica", 10)
#             p.drawString(50, y, f"YOUR RESPONSE: {q.user_answer or '[No response provided]'}")
#             y -= 14
#             p.drawString(50, y, f"AI FEEDBACK: {q.ai_feedback or '[No feedback available]'}")
#             y -= 14
#             p.drawString(50, y, f"SCORE: {q.score or 'N/A'}/10")
#             y -= 20

#         # Summary
#         if y < 100:
#             p.showPage()
#             y = height - 50

#         p.setFont("Helvetica-Bold", 12)
#         p.drawString(50, y, "INTERVIEW SUMMARY:")
#         y -= 15
#         p.setFont("Helvetica", 10)
#         for line in generate_interview_summary(session).splitlines():
#             p.drawString(50, y, line)
#             y -= 12

#         # Generated date
#         y -= 20
#         p.drawString(50, y, f"Generated on: {timezone.now().strftime('%B %d, %Y at %I:%M %p')}")

#         # Finalize PDF
#         p.showPage()
#         p.save()

#         buffer.seek(0)
#         response = HttpResponse(buffer, content_type='application/pdf')
#         response['Content-Disposition'] = f'attachment; filename="interview_results_{session_id}.pdf"'
#         return response

#     except InterviewSession.DoesNotExist:
#         return JsonResponse({'error': 'Session not found'}, status=404)
#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return JsonResponse({'error': f'Failed to generate PDF: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
@login_required(login_url='/login/')
def get_interview_history(request):
    """Get user's interview history"""
    try:
        sessions = InterviewSession.objects.filter(user=request.user).order_by('-timestamp')[:10]
        
        history_data = []
        for session in sessions:
            history_data.append({
                'session_id': str(session.session_id),
                'timestamp': session.timestamp.strftime('%B %d, %Y at %I:%M %p'),
                'input_type': session.get_user_input_type_display(),
                'is_completed': session.is_completed,
                'total_score': session.total_score,
                'questions_answered': session.questions.count()
            })
        
        return JsonResponse({'history': history_data})
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to get history: {str(e)}'}, status=500)


#                        langchain, CNN, django, rnn, python, scikit, pandas, tensorflow

# from google.cloud import translate_v2 as translate # Import the v2 client

# # Initialize Google Cloud Translation client (will pick up credentials from GOOGLE_APPLICATION_CREDENTIALS env var)
# translate_client = translate.Client()

# @csrf_exempt # Use this decorator for simplicity in development, but configure proper CSRF protection in production
# def translate_text_view(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             text = data.get('text')
#             target_language = data.get('target_language')

#             if not text or not target_language:
#                 return JsonResponse({'error': 'Missing text or target_language'}, status=400)

#             # Perform the translation
#             # The 'text' can be a list of strings if you want to translate multiple pieces at once
#             result = translate_client.translate(
#                 text,
#                 target_language=target_language
#             )

#             translated_text = result['translatedText']
#             return JsonResponse({'translatedText': translated_text})

#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON'}, status=400)
#         except Exception as e:
#             # Log the error in a real application
#             return JsonResponse({'error': str(e)}, status=500)
#     else:
#         return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)