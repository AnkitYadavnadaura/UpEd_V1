from django.contrib import admin
from .models import CustomUser, CoursePlan, Gen_Content, InterviewQuestion, InterviewSession, CoursePlanOnSyllabus
# Register your models here.
admin.site.register(CustomUser)
admin.site.register(CoursePlan)
admin.site.register(Gen_Content)
admin.site.register(InterviewQuestion)
admin.site.register(InterviewSession)
admin.site.register(CoursePlanOnSyllabus)