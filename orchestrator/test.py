import requests
import json
url = "http://172.25.0.4:8080/projects/andreas/digits-classification/reports/all/"
report = {'model':'1','description':'fdsfsdfs','report':'fhskjdfhjsdfs',}
r = requests.post(url,report)
print(r.status_code)

#url_model = "http://localhost:8080/projects/andreas/digits-classification/models/all/"
#model = {'name':'MNIST Global','description':'ffkljfs','url':'http://test.org'}
#r = requests.post(url_model,model)
#print(r.status_code)

