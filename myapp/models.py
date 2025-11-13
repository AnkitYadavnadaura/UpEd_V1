from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import JSONField
# Create your models here.

class CustomUser(AbstractUser):
    name = models.CharField(max_length=128)
    phone_number = models.CharField(max_length=20)
    email = models.EmailField(unique=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(
        max_length=50,
        choices = [("male", "Male"), ("female", "Female"), ("other", "Other")],
        default="other",
        null=True,
        blank=True
    )
    country = models.CharField(max_length=50, null=True, blank=True)
    language = models.CharField(max_length=100, null=True, blank=True, default='English')
    profile_picture = models.CharField(max_length=256, null=True, blank=True)
    profile_picture_text = models.TextField(null=True, blank=True)


class Gen_Content(models.Model):
    user = models.ForeignKey('myapp.CustomUser', on_delete=models.CASCADE, default=1)
    course_id = models.CharField(max_length=100, null=False, blank=False)
    topic_name = models.CharField(max_length=256, null=False, blank=False)
    subtopic_name = models.CharField(max_length=256, null=False, blank=False)
    notes = models.TextField(null=True, blank=True)
    quizzes = models.JSONField(null=True, blank=True)
    quiz_score = models.PositiveIntegerField(null=True, blank=True)
    quiz_status = models.CharField(max_length=50, null=True, blank=True)
    project = models.TextField(null=True, blank=True)
    project_score = models.PositiveIntegerField(null=True, blank=True)
    project_repo = models.URLField(null=True, blank=True)
    project_status = models.CharField(max_length=50, null=True, blank=True)
    ai_feedback = models.JSONField(blank=True, null=True)
    class Meta:
        unique_together = ('user', 'course_id', 'topic_name', 'subtopic_name')

class CoursePlan(models.Model):
    # One plan per user per course (enforced in code)
    user = models.ForeignKey('myapp.CustomUser', on_delete=models.CASCADE, related_name='course_plans')
    course_name = models.CharField(max_length=200)
    course_slug = models.SlugField(max_length=200)
    level = models.CharField(max_length=100)       # e.g., Beginner
    language = models.CharField(max_length=50)     # e.g., English
    goal = models.TextField(blank=True)
    status = models.CharField(max_length=50, default='In progress')
    score = models.IntegerField(default=0)
    max_score = models.IntegerField(default=0)
    completion_date = models.DateTimeField(null=True, blank=True)
    # Optionally, created_at/updated_at

    class Meta:
        unique_together = (('user', 'course_slug'),)

    def __str__(self):
        return f"{self.course_name} ({self.get_username_display()})"

    def get_username_display(self):
        return self.user.username if self.user else self.username

class Module(models.Model):
    course_plan = models.ForeignKey(CoursePlan, on_delete=models.CASCADE, related_name='modules')
    topic = models.CharField(max_length=255)
    topic_slug = models.SlugField(max_length=255)
    video_title = models.CharField(max_length=512, null=True, blank=True)
    video_link = models.URLField(max_length=512, null=True, blank=True)
    video_author = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.topic}"

class Subtopic(models.Model):
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='subtopics')
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255)
    

    def __str__(self):
        return self.name


from django.utils import timezone
import uuid

class InterviewSession(models.Model):
    INPUT_TYPE_CHOICES = [
        ('PDF', 'PDF Resume'),
        ('TEXT', 'Skills Text'),
    ]
    
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey('myapp.CustomUser', on_delete=models.CASCADE, default=1)
    timestamp = models.DateTimeField(default=timezone.now)
    user_input_type = models.CharField(max_length=10, choices=INPUT_TYPE_CHOICES)
    user_input_content = models.TextField()
    resume_pdf_data = models.BinaryField(null=True, blank=True)
    resume_pdf_name = models.CharField(max_length=255, null=True, blank=True)
    current_question_index = models.IntegerField(default=0)
    is_completed = models.BooleanField(default=False)
    total_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.user_id} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

class InterviewQuestion(models.Model):
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name='questions')
    question_number = models.IntegerField()
    question_text = models.TextField()
    user_answer = models.TextField(blank=True, null=True)
    ai_feedback = models.TextField(blank=True, null=True)
    score = models.FloatField(null=True, blank=True)  # Score for this question (1-10)
    timestamp_asked = models.DateTimeField(default=timezone.now)
    timestamp_answered = models.DateTimeField(null=True, blank=True)
    needs_clarification = models.BooleanField(default=False)
    clarification_prompt = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['question_number']
    
    def __str__(self):
        return f"Q{self.question_number} - Session {self.session.session_id}"

class CoursePlanOnSyllabus(models.Model):
    user =models.ForeignKey('myapp.CustomUser', on_delete=models.CASCADE, default=1)
    syllabus = models.BinaryField()
    syllabus_text = models.TextField()
    resources = models.BinaryField(blank=True, null=True)
    resources_text = models.TextField(blank=True, null=True)
    course_name = models.CharField(max_length=200)
    course_slug = models.SlugField(max_length=200)
    status = models.CharField(max_length=50, default='In progress')
    score = models.IntegerField(default=0)
    max_score = models.IntegerField(default=0)
    completion_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = (('user', 'course_slug'),)

    def __str__(self):
        return f"{self.course_name} ({self.get_username_display()})"

    def get_username_display(self):
        return self.user.username if self.user else self.username

class ModuleOnSyllabus(models.Model):
    course_plan = models.ForeignKey(CoursePlanOnSyllabus, on_delete=models.CASCADE, related_name='modules')
    topic = models.CharField(max_length=255)
    topic_slug = models.SlugField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.topic}"

class SubtopicOnSyllabus(models.Model):
    module = models.ForeignKey(ModuleOnSyllabus, on_delete=models.CASCADE, related_name='subtopics')
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255)
    

    def __str__(self):
        return self.name
