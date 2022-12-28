from unittest import TestCase 
from unittest.mock import patch, Mock, call 
from samtranslator.parser.parser import Parser 
from samtranslator.plugins import LifeCycleEvents 
from samtranslator.model.exceptions import InvalidDocumentationException, InvalidTemplateException,InvalidResourceException

class TestParser(TestCase):
  
