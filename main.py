# Test case
import json

test_json = '[{\"childName\":\"Adebanjo Mayowa\",\"email\":\"temidun2003ade@gmail.com\",\"phone\":\"+13828822712\",\"age\":5,\"gender\":\"Male\",\"connections\":\"asasas\",\"details\":\"asasas\",\"hobbies\":\"saasas\"}]'
cleaned_json = test_json.encode().decode('unicode_escape')
parsed_data = json.loads(cleaned_json)
print("Test parse result:", parsed_data)