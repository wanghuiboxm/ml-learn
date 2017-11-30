#! /usr/bin/env python
# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import HttpResponse
import json


def hello(request):
    rsp = {"code": 200, "message": "成功"}
    return HttpResponse(json.dumps(rsp, ensure_ascii=False), content_type="application/json")