"""
URL configuration for pep project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path("register/", views.register, name="register"),
    path("verify_otp/<str:email>/", views.verify_otp, name="verify_otp"),
    path("resend_otp/<str:email>/", views.resend_otp, name="resend_otp"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path('check_email/', views.check_email, name='check_email'),
    path("login/", views.loginuser, name="login"),
    path("logout/", views.logoutuser, name="logout"),
    path("submit_preferences/", views.submit_preferences, name = 'submit_preferences'),
    path("courses/<course_slug>/", views.course_page, name="course_page"),
    path("courses/<course_slug>/<topic_slug>/", views.topic_page, name = "topic_page"),
    path("courses/<course_slug>/<topic_slug>/<subtopic_slug>/", views.subtopic_page, name="subtopic_page"),
    path('submit_quiz_score/', views.submit_quiz_score, name='submit_quiz_score'),
    path('submit_project_score/', views.submit_project_score, name='submit_project_score'),
    path('certificate/<course_slug>/', views.certificate_gen, name='certificate_page'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('', views.index, name='index'),
    path('interview/', views.interview, name='interview'),
    path('start_interview/', views.start_interview, name='start_interview'),
    path('submit_answer/', views.submit_answer, name='submit_answer'),
    path('download_results/<uuid:session_id>/', views.download_results, name='download_results'),
    path('interview_history/', views.get_interview_history, name='interview_history'),
    path("submit_syllabus_and_resources/", views.submit_syllabus_and_resources, name = 'submit_syllabus_and_resources'),
    path("getcourseslist/<e>", views.getcourseslist, name = 'getcourseslist'),
    path("courses/<course_slug>/<topic_slug>/<subtopic_slug>/chat/", views.subtopic_chat, name="submit_feedback"),
    path('password_reset/', 
         auth_views.PasswordResetView.as_view(
             template_name='password_reset.html',
             email_template_name='password_reset_email.html',
             subject_template_name='password_reset_subject.txt'
         ),
         name='password_reset'),

    # Page after email sent
    path('password_reset/done/',
         auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'),
         name='password_reset_done'),

    # Link that user clicks from email
    path('reset/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'),
         name='password_reset_confirm'),

    # Password reset complete page
    path('reset/done/',
         auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'),
         name='password_reset_complete'),
    # path('translate/', views.translate_text_view, name='translate_text'),
]
